import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from tqdm import tqdm

# Constants
INPUT_SIZE = 64
HIDDEN_SIZE = 500
NUM_LAYERS = 5
NUM_CLASSES = 41  # Number of phonemes
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 1e-5

# PhoDe Model
class PhoDe(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PhoDe, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.bn = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out)
        return out

# Dataset
class LibriSpeechDataset(Dataset):
    def __init__(self, root_dir, subset='train-clean-100', transform=None):
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        for speaker in os.listdir(self.root_dir):
            speaker_dir = os.path.join(self.root_dir, speaker)
            for chapter in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter)
                for file in os.listdir(chapter_dir):
                    if file.endswith('.flac'):
                        audio_path = os.path.join(chapter_dir, file)
                        transcript_path = os.path.join(chapter_dir, file[:-5] + '.txt')
                        with open(transcript_path, 'r') as f:
                            transcript = f.read().strip()
                        self.data.append((audio_path, transcript))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, transcript = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if self.transform:
            waveform = self.transform(waveform)

        # Convert transcript to phonemes (placeholder - replace with actual phoneme conversion)
        phonemes = [ord(c) - ord('a') for c in transcript.lower() if c.isalpha()]
        
        return waveform, torch.tensor(phonemes)

# Audio processing functions
def create_vocoder(num_channels=16):
    return torchaudio.transforms.Resample(orig_freq=16000, new_freq=num_channels)

def apply_vocoder(audio, vocoder):
    return vocoder(audio)

def add_noise(audio, snr_db):
    signal_power = audio.norm(p=2)
    noise_power = signal_power / (10 ** (snr_db / 20))
    noise = torch.randn_like(audio) * noise_power
    return audio + noise

def spectogram(audio, n_fft=400, hop_length=160):
    return torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(audio)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    transform = torchaudio.transforms.Compose([
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=INPUT_SIZE),
        spectogram
    ])
    
    train_dataset = LibriSpeechDataset(root_dir='path/to/LibriSpeech', subset='train-clean-100', transform=transform)
    test_dataset = LibriSpeechDataset(root_dir='path/to/LibriSpeech', subset='test-clean', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create and train the model
    model = PhoDe(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'phode_model.pth')

if __name__ == "__main__":
    main()