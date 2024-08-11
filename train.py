import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, HubertConfig
from sklearn.cluster import KMeans
import numpy as np
import math
from typing import Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.cluster import MiniBatchKMeans

if torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

class SpeechToSpeechDataset(Dataset):
    def __init__(self, 
                 audio_dir: str, 
                 sample_rate: int = 16000, 
                 max_audio_length: int = 16000):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        
        if waveform.shape[1] < self.max_audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_audio_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_audio_length]
        
        return waveform, waveform

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_audio, target_audio = zip(*batch)
    input_audio = torch.stack(input_audio)
    target_audio = torch.stack(target_audio)
    return input_audio, target_audio

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class SpeechToSpeechLM(nn.Module):
    def __init__(self, num_centroids: int, hidden_size: int, num_layers: int, dropout: float = 0.01):
        super(SpeechToSpeechLM, self).__init__()
        self.hubert = HubertModel(HubertConfig())
        self.kmeans = MiniBatchKMeans(n_clusters=num_centroids, batch_size=256)
        self.embedding = nn.Embedding(num_centroids, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, num_centroids)


    def process_audio(self, audio_input: torch.Tensor) -> torch.Tensor:
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(1)
        elif audio_input.dim() == 4:
            audio_input = audio_input.squeeze(1).squeeze(1)

        min_audio_length = 16000
        if audio_input.size(1) < min_audio_length:
            padding = min_audio_length - audio_input.size(1)
            audio_input = torch.nn.functional.pad(audio_input, (0, padding))

        features = self.hubert(audio_input).last_hidden_state
        return features

    def forward(self, audio_input: torch.Tensor, target_audio: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.process_audio(audio_input)
        
        # Reshape features for clustering
        features_2d = features.view(-1, features.size(-1)).detach().cpu().numpy()
        
        if self.training:
            self.kmeans = self.kmeans.partial_fit(features_2d)
        
        tokens = torch.tensor(self.kmeans.predict(features_2d), device=audio_input.device, dtype=torch.long)
        tokens = tokens.view(features.size(0), features.size(1))
        
        embedded = self.embedding(tokens)
        embedded = self.positional_encoding(embedded)
        
        max_sequence_length = 1024
        if embedded.size(1) > max_sequence_length:
            encoder_outputs = []
            for i in range(0, embedded.size(1), max_sequence_length):
                chunk = embedded[:, i:i+max_sequence_length]
                output = self.encoder(chunk)
                encoder_outputs.append(output)
            encoder_output = torch.cat(encoder_outputs, dim=1)
        else:
            encoder_output = self.encoder(embedded)
        
        if self.training and target_audio is not None:
            target_features = self.process_audio(target_audio)
            target_features_2d = target_features.view(-1, target_features.size(-1)).detach().cpu().numpy()
            target_tokens = torch.tensor(self.kmeans.predict(target_features_2d), 
                                         device=audio_input.device, dtype=torch.long)
            target_tokens = target_tokens.view(target_features.size(0), target_features.size(1))
            
            target_embedded = self.embedding(target_tokens)
            target_embedded = self.positional_encoding(target_embedded)
            
            decoder_output = self.decoder(target_embedded, encoder_output)
        else:
            # During inference, we need to generate output step by step
            decoder_input = torch.zeros_like(encoder_output[:, :1])
            decoder_outputs = []
            for i in range(encoder_output.size(1)):
                step_output = self.decoder(decoder_input, encoder_output)
                decoder_outputs.append(step_output)
                decoder_input = torch.cat([decoder_input, step_output[:, -1:]], dim=1)
            decoder_output = torch.cat(decoder_outputs, dim=1)
        
        logits = self.fc(decoder_output)
        return logits

def create_dataloader(audio_dir: str, 
                      batch_size: int = 32, 
                      num_workers: int = 4, 
                      sample_rate: int = 16000, 
                      max_audio_length: int = 16000) -> DataLoader:
    dataset = SpeechToSpeechDataset(audio_dir, sample_rate, max_audio_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )



if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Reduce the number of centroids to match the dataset size
        num_centroids = 49  # Changed from 100 to 50

        model = SpeechToSpeechLM(num_centroids=num_centroids, hidden_size=768, num_layers=1).to(device)

        dataloader = create_dataloader(
            audio_dir="/mnt/e/Speech-to-Speech-LM/data",
            batch_size=8,  # Increased from 1 to 8
            num_workers=4,
            sample_rate=16000,
            max_audio_length=16000
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in dataloader:
                input_audio, target_audio = batch
                input_audio, target_audio = input_audio.to(device), target_audio.to(device)
                
                optimizer.zero_grad()
                
                output = model(input_audio, target_audio)
                
                with torch.no_grad():
                    target_features = model.process_audio(target_audio)
                    target_features_2d = target_features.view(-1, target_features.size(-1)).cpu().numpy()
                    target_tokens = torch.tensor(model.kmeans.predict(target_features_2d), 
                                                 device=device, dtype=torch.long)
                    target_tokens = target_tokens.view(target_features.size(0), target_features.size(1))
                
                loss = criterion(output.view(-1, output.size(-1)), target_tokens.view(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "speech_to_speech_model.pth")

    except Exception as e:
        print(f"An error occurred: {e}")
