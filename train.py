import torch
import torch.nn as nn
import torchaudio
import random
from transformers import HubertModel, HubertConfig
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import math
from typing import Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader
import os
import logging
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class SpeechToSpeechDataset(Dataset):
    def __init__(self, 
                 audio_dir: str, 
                 sample_rate: int = 16000, 
                 max_audio_length: int = 16000,
                 augment: bool = False):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.augment = augment
        self.audio_files = [
            f for f in os.listdir(audio_dir) 
            if f.endswith('.wav') or f.endswith('.mp3')
        ]
        logging.info(f"Found {len(self.audio_files)} audio files.")
            
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Data Augmentation: Apply random noise and time stretching
        if self.augment:
            waveform = self.apply_augmentation(waveform)
        
        # Convert to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)
        
        # Pad or trim the waveform to the desired length
        if waveform.shape[1] < self.max_audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_audio_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_audio_length]
        
        return waveform, waveform
    
    def apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        # Randomly apply noise
        if random.random() < 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Randomly apply time stretching
        if random.random() < 0.5:
            stretch_factor = random.uniform(0.8, 1.2)
            waveform = self.time_stretch(waveform, stretch_factor)
        
        return waveform
    
    def time_stretch(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        # Use torchaudio's time stretching
        try:
            waveform = torchaudio.transforms.TimeStretch()(waveform)
            # Note: TimeStretch requires complex input from a spectrogram. Implementing a simple placeholder.
            # For simplicity, this is left as a no-op. Implement proper time stretching as needed.
        except Exception as e:
            logging.warning(f"Time stretching failed: {e}")
        return waveform

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_audio, target_audio = zip(*batch)
    input_audio = torch.stack(input_audio)
    target_audio = torch.stack(target_audio)
    return input_audio, target_audio

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].unsqueeze(0).to(x.device)
        return x

class SpeechToSpeechLM(nn.Module):
    def __init__(self, num_centroids: int, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super(SpeechToSpeechLM, self).__init__()
        self.hubert = HubertModel(HubertConfig())
        self.embedding = nn.Embedding(num_centroids, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, num_centroids)
    
    def forward(self, audio_input: torch.Tensor, 
                target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:

        features = self.hubert(audio_input).last_hidden_state  # Shape: [batch, seq_len, hidden_size]
        
        embedded = self.embedding(target_tokens)  # Shape: [batch, seq_len, hidden_size]
        embedded = self.positional_encoding(embedded)
        
        encoder_output = self.encoder(features)  # Shape: [batch, seq_len, hidden_size]
        
        if self.training and target_tokens is not None:
            decoder_output = self.decoder(embedded, encoder_output)  # Shape: [batch, seq_len, hidden_size]
        else:
            # Inference mode: autoregressive decoding
            batch_size, seq_len, _ = encoder_output.size()
            generated_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=audio_input.device)
            generated_embedding = self.embedding(generated_tokens)
            generated_embedding = self.positional_encoding(generated_embedding)
            
            for _ in tqdm(range(seq_len), desc="Generating tokens"):
                decoder_output = self.decoder(generated_embedding, encoder_output)
                logits = self.fc(decoder_output[:, -1, :])  # Shape: [batch, num_centroids]
                predicted_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: [batch, 1]
                generated_tokens = torch.cat([generated_tokens, predicted_tokens], dim=1)
                new_embedding = self.embedding(predicted_tokens)
                new_embedding = self.positional_encoding(new_embedding)
                generated_embedding = torch.cat([generated_embedding, new_embedding], dim=1)
            
            return generated_tokens  # Returning token sequence during inference
        
        logits = self.fc(decoder_output)  # Shape: [batch, seq_len, num_centroids]
        return logits

def create_dataloader(audio_dir: str, 
                      batch_size: int = 32, 
                      num_workers: int = 4, 
                      sample_rate: int = 16000, 
                      max_audio_length: int = 16000,
                      split: float = 0.8,
                      augment: bool = False) -> Tuple[DataLoader, DataLoader]:

    dataset = SpeechToSpeechDataset(audio_dir, sample_rate, max_audio_length, augment=augment)
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logging.info(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

def precompute_kmeans(audio_dir: str, 
                     sample_rate: int, 
                     max_audio_length: int, 
                     num_centroids: int,
                     batch_size: int = 32, 
                     num_workers: int = 4) -> MiniBatchKMeans:

    dataset = SpeechToSpeechDataset(audio_dir, sample_rate, max_audio_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    hubert = HubertModel(HubertConfig()).to(device)
    hubert.eval()
    
    all_features = []
    with torch.no_grad():
        for input_audio, _ in tqdm(dataloader, desc="Extracting features for KMeans"):
            input_audio = input_audio.to(device)
            features = hubert(input_audio).last_hidden_state  # Shape: [batch, seq_len, hidden_size]
            all_features.append(features.view(-1, features.size(-1)).cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)  # Shape: [total_seq_len, hidden_size]
    logging.info(f"Total features for KMeans: {all_features.shape}")
    
    kmeans = MiniBatchKMeans(n_clusters=num_centroids, batch_size=256, verbose=1, random_state=42)
    kmeans.fit(all_features)
    logging.info("KMeans clustering completed.")
    
    return kmeans

def save_kmeans(kmeans: MiniBatchKMeans, filepath: str):
    np.save(filepath, kmeans.cluster_centers_)
    logging.info(f"KMeans cluster centers saved to {filepath}")

def load_kmeans(filepath: str) -> MiniBatchKMeans:
    centers = np.load(filepath)
    kmeans = MiniBatchKMeans(n_clusters=centers.shape[0], random_state=42)
    kmeans.cluster_centers_ = centers
    logging.info(f"KMeans cluster centers loaded from {filepath}")
    return kmeans

def initialize_model(num_centroids: int, hidden_size: int, num_layers: int) -> SpeechToSpeechLM:
    model = SpeechToSpeechLM(num_centroids=num_centroids, hidden_size=hidden_size, num_layers=num_layers).to(device)
    logging.info("Model initialized.")
    return model

def save_model(model: SpeechToSpeechLM, kmeans: MiniBatchKMeans, model_path: str, kmeans_path: str):
    torch.save(model.state_dict(), model_path)
    save_kmeans(kmeans, kmeans_path)
    logging.info(f"Model saved to {model_path} and KMeans to {kmeans_path}")

def load_model(model: SpeechToSpeechLM, model_path: str, kmeans_path: str) -> Tuple[SpeechToSpeechLM, MiniBatchKMeans]:
    model.load_state_dict(torch.load(model_path, map_location=device))
    kmeans = load_kmeans(kmeans_path)
    model.to(device)
    logging.info(f"Model loaded from {model_path} and KMeans from {kmeans_path}")
    return model, kmeans

def initialize_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = 'ReduceLROnPlateau', **kwargs):
    if scheduler_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def main():
    # Configuration parameters
    audio_dir = "/mnt/e/Speech-to-Speech-LM/data"
    sample_rate = 16000
    max_audio_length = 16000
    num_centroids = 50 
    hidden_size = 768
    num_layers = 2
    dropout = 0.1
    batch_size = 8
    num_workers = 4
    num_epochs = 50  # Increased to allow early stopping
    learning_rate = 1e-4
    model_path = "speech_to_speech_model.pth"
    kmeans_path = "kmeans_centers.npy"
    patience = 5  # For early stopping
    min_delta = 0.001  # Minimum improvement for early stopping
    gradient_clip = 1.0  # Max norm for gradient clipping
    scheduler_type = 'ReduceLROnPlateau'
    scheduler_params = {'mode': 'min', 'factor': 0.5, 'patience': 2, 'verbose': True}
    
    # Initialize or load KMeans
    if not os.path.exists(kmeans_path):
        logging.info("Precomputing KMeans clustering...")
        kmeans = precompute_kmeans(
            audio_dir=audio_dir,
            sample_rate=sample_rate,
            max_audio_length=max_audio_length,
            num_centroids=num_centroids,
            batch_size=batch_size,
            num_workers=num_workers
        )
        save_kmeans(kmeans, kmeans_path)
    else:
        logging.info("Loading existing KMeans clustering...")
        kmeans = load_kmeans(kmeans_path)

    # Initialize or load model
    model = initialize_model(num_centroids=num_centroids, hidden_size=hidden_size, num_layers=num_layers)

    if os.path.exists(model_path):
        logging.info("Loading existing model state...")
        model, kmeans = load_model(model, model_path, kmeans_path)

    # Create DataLoaders with data augmentation for training set
    train_loader, val_loader = create_dataloader(
        audio_dir=audio_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
        augment=True  # Enable data augmentation for training
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize learning rate scheduler
    scheduler = initialize_scheduler(optimizer, scheduler_type, **scheduler_params)

    # Early Stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "best_speech_to_speech_model.pth"

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch in progress_bar:
            input_audio, target_audio = batch
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)
            
            # Forward pass
            features = model.hubert(input_audio).last_hidden_state  # [batch, seq_len, hidden_size]
            features_2d = features.view(-1, features.size(-1)).cpu().numpy()  # For KMeans
            tokens = torch.tensor(kmeans.predict(features_2d), device=device, dtype=torch.long)
            tokens = tokens.view(features.size(0), features.size(1))
            
            optimizer.zero_grad()
            logits = model(input_audio, target_tokens=tokens)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), tokens.view(-1))
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=-1)
            mask = tokens != -100
            correct = (preds == tokens) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), accuracy=100.0 * total_correct / max(total_tokens, 1))
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100.0 * total_correct / max(total_tokens, 1)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        
        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_tokens = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_audio, target_audio = batch
                input_audio = input_audio.to(device)
                target_audio = target_audio.to(device)
                
                features = model.hubert(input_audio).last_hidden_state
                features_2d = features.view(-1, features.size(-1)).cpu().numpy()
                tokens = torch.tensor(kmeans.predict(features_2d), device=device, dtype=torch.long)
                tokens = tokens.view(features.size(0), features.size(1))
                
                logits = model(input_audio, target_tokens=tokens)
                loss = criterion(logits.view(-1, logits.size(-1)), tokens.view(-1))
                val_loss += loss.item()
                
                # Compute accuracy
                preds = torch.argmax(logits, dim=-1)
                mask = tokens != -100
                correct = (preds == tokens) & mask
                val_correct += correct.sum().item()
                val_tokens += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / max(val_tokens, 1)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Step the scheduler
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        # Check for improvement
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(model, kmeans, best_model_path, kmeans_path)
            logging.info(f"Validation loss decreased. Saving new best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered.")
                break
        
        # Save the latest model state
        save_model(model, kmeans, model_path, kmeans_path)
    
    logging.info("Training completed.")
    logging.info(f"Best validation loss achieved: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
