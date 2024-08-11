import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel
from sklearn.cluster import KMeans
import numpy as np
import math
from typing import Optional, List

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def chunk_audio(audio: torch.Tensor, chunk_size: int = 30000, stride: int = 15000) -> torch.Tensor:
    audio = audio.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    chunks = audio.unfold(0, chunk_size, stride)
    return chunks.unsqueeze(1).unsqueeze(1)  # Add batch and channel dimensions

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
    def __init__(self, hubert_model: HubertModel, kmeans_model: KMeans, 
                 num_centroids: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.1):
        super(SpeechToSpeechLM, self).__init__()
        self.hubert = hubert_model
        self.kmeans = kmeans_model
        self.embedding = nn.Embedding(num_centroids, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, 
                                                    dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_centroids)

    def process_audio(self, audio_input: torch.Tensor) -> torch.Tensor:
        chunk_size = 30000  # Adjust this based on your HuBERT model's requirements
        stride = 15000  # Adjust this for desired overlap

        chunks = chunk_audio(audio_input, chunk_size, stride)
        all_tokens = []

        with torch.no_grad():
            for chunk in chunks:
                features = self.hubert(chunk).last_hidden_state
                tokens = self.kmeans.predict(features.detach().cpu().numpy())
                all_tokens.extend(tokens)

        return torch.tensor(all_tokens, device=audio_input.device, dtype=torch.long)

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        tokens = self.process_audio(audio_input)
        embedded = self.embedding(tokens)
        embedded = self.positional_encoding(embedded)
        
        max_sequence_length = 1024  # Adjust based on your transformer's capacity
        if embedded.size(1) > max_sequence_length:
            outputs = []
            for i in range(0, embedded.size(1), max_sequence_length):
                chunk = embedded[:, i:i+max_sequence_length]
                output = self.transformer(chunk)
                outputs.append(output)
            output = torch.cat(outputs, dim=1)
        else:
            output = self.transformer(embedded)
        
        logits = self.fc(output)
        return logits

    def generate(self, audio_input: torch.Tensor, max_length: int, 
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> List[int]:
        device = audio_input.device
        generated = []

        with torch.no_grad():
            tokens = self.process_audio(audio_input)
            input_embedded = self.embedding(tokens)
            input_embedded = self.positional_encoding(input_embedded)
            
            max_sequence_length = 1024  # Should match the value in forward method
            
            for _ in range(max_length):
                if input_embedded.size(1) > max_sequence_length:
                    context = input_embedded[:, -max_sequence_length:]
                else:
                    context = input_embedded

                output = self.transformer(context)
                logits = self.fc(output[:, -1, :])
                logits = logits / temperature
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)

                next_embedded = self.embedding(torch.tensor([next_token], device=device)).unsqueeze(0)
                next_embedded = self.positional_encoding(next_embedded)
                input_embedded = torch.cat([input_embedded, next_embedded], dim=1)

                if len(generated) >= max_length:
                    break

        return generated

    def train_step(self, audio_input: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        logits = self(audio_input)
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        return loss

    def tokens_to_features(self, tokens: List[int]) -> np.ndarray:
        return self.kmeans.cluster_centers_[tokens]

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'kmeans_model': self.kmeans,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, hubert_model: HubertModel):
        checkpoint = torch.load(path)
        kmeans_model = checkpoint['kmeans_model']
        num_centroids = kmeans_model.n_clusters
        hidden_size = checkpoint['model_state_dict']['embedding.weight'].size(1)
        num_layers = sum(1 for k in checkpoint['model_state_dict'] if k.startswith('transformer.layers'))

        model = cls(hubert_model, kmeans_model, num_centroids, hidden_size, num_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

def preprocess_audio(audio_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.mean(dim=0, keepdim=True)  # Convert to mono if stereo
        if sample_rate != target_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
        return audio.unsqueeze(0)  # Shape: (1, 1, audio_length)
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None
