import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import seaborn as sns
from scipy.fft import fft
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from tabulate import tabulate
import time
from scipy.stats import pearsonr
import torch.nn.functional as F

# Configuration
CONFIG = {
    'data_path': 'Breast_Cancer.csv',
    'batch_size': 32,
    'n_epochs': 100,
    'learning_rate': 0.001,
    'input_size': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'k_folds': 5  # Number of folds for cross-validation
}

# Power Quality Disturbance Classes
DISTURBANCE_TYPES = {
    0: 'Malignant',
    1: 'Benign'
}

CONFIG['num_classes'] = len(DISTURBANCE_TYPES)

# ================== DATA LOADING & PREPROCESSING ================== #
def load_and_preprocess_data():
    """Load and preprocess power quality disturbance dataset"""
    df = pd.read_csv(CONFIG['data_path'])
    
    signals = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    if np.min(labels) > 0:
        labels = labels - 1
    
    signals = 2 * (signals - np.min(signals, axis=1, keepdims=True)) / \
             (np.max(signals, axis=1, keepdims=True) - np.min(signals, axis=1, keepdims=True)) - 1
    
    signals_tensor = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return signals_tensor, labels_tensor

# ================== NEURAL NETWORK MODELS ================== #
class BasicBlock1D(nn.Module):
    """Basic residual block for 1D signals"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    """ResNet-based 1D CNN for power quality disturbance classification"""
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, CONFIG['num_classes'])
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock1D(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN(nn.Module):
    """CNN-based power quality disturbance classifier"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, CONFIG['num_classes'])
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class HybridAutoencoderClassifier(nn.Module):
    """Autoencoder + Classifier for joint reconstruction and classification"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, CONFIG['num_classes'])
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        flattened = encoded.view(encoded.size(0), -1)
        classification = self.classifier(flattened)
        return reconstructed, classification

class InceptionModule1D(nn.Module):
    """Inception module for 1D signals with consistent output sizes"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_bottleneck = nn.BatchNorm1d(out_channels)
        
        # Use same padding calculations to ensure consistent output sizes
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=10, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=20, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=40, padding='same', bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pool = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_bottleneck = self.relu(self.bn_bottleneck(self.bottleneck(x)))
        
        branch1 = self.relu(self.bn1(self.conv1(x_bottleneck)))
        branch2 = self.relu(self.bn2(self.conv2(x_bottleneck)))
        branch3 = self.relu(self.bn3(self.conv3(x_bottleneck)))
        branch_pool = self.relu(self.bn_pool(self.conv_pool(self.maxpool(x))))
        
        out = torch.cat([branch1, branch2, branch3, branch_pool], dim=1)
        return out

class InceptionTime(nn.Module):
    """InceptionTime model for time series classification with consistent sizes"""
    def __init__(self):
        super().__init__()
        self.inception1 = InceptionModule1D(1, 32)
        self.inception2 = InceptionModule1D(128, 32)
        self.inception3 = InceptionModule1D(128, 32)
        
        # Residual connections with proper sizing
        self.residual1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(128)
        )
        self.residual2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(128)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, CONFIG['num_classes'])
    
    def forward(self, x):
        # First inception block
        res = self.residual1(x)
        x = self.inception1(x)
        # Ensure same length before adding
        min_len = min(x.size(2), res.size(2))
        x = x[:, :, :min_len]
        res = res[:, :, :min_len]
        x = x + res
        x = nn.functional.relu(x)
        
        # Second inception block
        res = self.residual2(x)
        x = self.inception2(x)
        # Ensure same length before adding
        min_len = min(x.size(2), res.size(2))
        x = x[:, :, :min_len]
        res = res[:, :, :min_len]
        x = x + res
        x = nn.functional.relu(x)
        
        # Third inception block
        x = self.inception3(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x

class TransformerClassifier(nn.Module):
    """Transformer-based classifier for time series"""
    def __init__(self):
        super().__init__()
        d_model = 64
        nhead = 4
        dim_feedforward = 128
        num_layers = 3
        dropout = 0.1
        
        self.embedding = nn.Conv1d(1, d_model, kernel_size=7, padding=3, stride=2)
        self.bn_embed = nn.BatchNorm1d(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, CONFIG['num_classes'])
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.bn_embed(x)
        x = nn.functional.relu(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.avgpool(x).squeeze(-1)
        x = self.fc(x)
        return x

class TimeSeriesTransformer(nn.Module):
    """Transformer-based model for time series classification"""
    def __init__(self):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 3
        self.dropout = 0.1
        
        # Patch embedding
        self.patch_size = 5
        self.patch_embed = nn.Conv1d(
            1, self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Positional encoding
        self.position_embedding = PositionalEncoding(self.embed_dim, dropout=self.dropout)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=256,
            dropout=self.dropout,
            activation='gelu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, CONFIG['num_classes'])
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, num_patches)
        x = x.permute(2, 0, 1)  # (num_patches, batch, embed_dim)
        
        # Add positional encoding
        x = self.position_embedding(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling and classification
        x = x.mean(dim=0)  # Average over sequence length
        x = self.norm(x)
        x = self.fc(x)
        return x

class XCM(nn.Module):
    """Explainable Convolutional Model for time series classification"""
    def __init__(self):
        super().__init__()
        # Temporal branch
        self.temporal_conv1 = nn.Conv1d(1, 32, kernel_size=8, padding='same')
        self.temporal_bn1 = nn.BatchNorm1d(32)
        self.temporal_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.temporal_bn2 = nn.BatchNorm1d(64)
        self.temporal_conv3 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.temporal_bn3 = nn.BatchNorm1d(128)
        
        # Feature branch
        self.feature_conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.feature_bn1 = nn.BatchNorm1d(32)
        self.feature_conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.feature_bn2 = nn.BatchNorm1d(64)
        self.feature_conv3 = nn.Conv1d(64, 128, kernel_size=1)
        self.feature_bn3 = nn.BatchNorm1d(128)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Classification head
        self.fc = nn.Linear(256, CONFIG['num_classes'])
    
    def forward(self, x):
        # Temporal branch
        t = F.relu(self.temporal_bn1(self.temporal_conv1(x)))
        t = F.relu(self.temporal_bn2(self.temporal_conv2(t)))
        t = F.relu(self.temporal_bn3(self.temporal_conv3(t)))
        
        # Feature branch
        f = F.relu(self.feature_bn1(self.feature_conv1(x)))
        f = F.relu(self.feature_bn2(self.feature_conv2(f)))
        f = F.relu(self.feature_bn3(self.feature_conv3(f)))
        
        # Concatenate branches
        x = torch.cat([t, f], dim=1)
        
        # Attention mechanism
        attn_weights = self.attention(x)
        x = torch.sum(x * attn_weights, dim=2)
        
        # Classification
        x = self.fc(x)
        return x

class DenseNet1D(nn.Module):
    """1D DenseNet for time series classification"""
    def __init__(self):
        super().__init__()
        self.growth_rate = 32
        self.block_config = (4, 4, 4)
        self.num_init_features = 64
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv1d(1, self.num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=self.growth_rate,
                kernel_size=3
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * self.growth_rate
            
            if i != len(self.block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        
        # Classification head
        self.classifier = nn.Linear(num_features, CONFIG['num_classes'])
        
        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool1d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class _DenseLayer(nn.Sequential):
    """Single layer of a DenseBlock"""
    def __init__(self, num_input_features, growth_rate, kernel_size):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv1d(
            num_input_features, growth_rate,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        ))
        self.add_module('dropout', nn.Dropout(0.2))
    
    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """DenseBlock consisting of multiple DenseLayers"""
    def __init__(self, num_layers, num_input_features, growth_rate, kernel_size):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                kernel_size
            )
            self.add_module(f'denselayer{i+1}', layer)

class _Transition(nn.Sequential):
    """Transition layer between DenseBlocks"""
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(
            num_input_features, num_output_features,
            kernel_size=1, stride=1, bias=False
        ))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))

class ConvLSTM(nn.Module):
    """Convolutional LSTM model for time series classification"""
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(256)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.fc = nn.Linear(256, CONFIG['num_classes'])
    
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Permute for LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        x = self.fc(context)
        return x
    
# ================== TRAINING & EVALUATION ================== #
class ModelTrainer:
    def __init__(self):
        self.metrics = {}
        self.training_times = {}
        self.model_performances = []
        self.kfold_metrics = {}  # To store metrics for each fold
    
    def train_model(self, model, train_loader, val_loader, model_name, fold=None):
        model.to(CONFIG['device'])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        
        best_val_acc = 0
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        
        start_time = time.time()
        
        for epoch in range(CONFIG['n_epochs']):
            model.train()
            running_loss = 0.0
            
            for signals, labels in train_loader:
                signals = signals.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                
                optimizer.zero_grad()
                
                if isinstance(model, HybridAutoencoderClassifier):
                    _, outputs = model(signals)
                    loss = criterion(outputs, labels)
                else:
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals = signals.to(CONFIG['device'])
                    labels = labels.to(CONFIG['device'])
                    
                    if isinstance(model, HybridAutoencoderClassifier):
                        _, outputs = model(signals)
                    else:
                        outputs = model(signals)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            epoch_loss = running_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            train_loss_history.append(epoch_loss)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_name = f'best_{model_name}_fold{fold}.pth' if fold is not None else f'best_{model_name}.pth'
                torch.save(model.state_dict(), save_name)
            
            if fold is not None:  # Only print detailed info for single fold training
                print(f'Fold {fold} Epoch [{epoch+1}/{CONFIG["n_epochs"]}] - '
                      f'Train Loss: {epoch_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_acc:.2f}%')
        
        training_time = time.time() - start_time
        if fold is not None:
            self.training_times[f'{model_name}_fold{fold}'] = training_time
        else:
            self.training_times[model_name] = training_time
        
        return train_loss_history, val_loss_history, val_acc_history
    
    def cross_validate(self, model, dataset, model_name):
        """Perform k-fold cross validation for a model"""
        kfold = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
        fold_results = []
        
        print(f"\n===== Starting {CONFIG['k_folds']}-Fold Cross Validation for {model_name} =====")
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print(f"\nFold {fold + 1}/{CONFIG['k_folds']}")
            
            # Split into train and test for this fold
            train_subsampler = Subset(dataset, train_ids)
            test_subsampler = Subset(dataset, test_ids)
            
            # Further split train into train and validation (80-20)
            val_size = int(0.2 * len(train_subsampler))
            train_size = len(train_subsampler) - val_size
            
            train_dataset, val_dataset = random_split(
                train_subsampler, [train_size, val_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
            test_loader = DataLoader(test_subsampler, batch_size=CONFIG['batch_size'])
            
            # Initialize new model for each fold
            model_instance = model.__class__()
            model_instance.to(CONFIG['device'])
            
            # Train and evaluate
            self.train_model(model_instance, train_loader, val_loader, model_name, fold=fold+1)
            model_instance.load_state_dict(torch.load(f'best_{model_name}_fold{fold+1}.pth'))
            self.evaluate_model(model_instance, test_loader, model_name, fold=fold+1)
        
        # Calculate and print average metrics across all folds
        if model_name in self.kfold_metrics:
            avg_accuracy = np.mean(self.kfold_metrics[model_name]['accuracy'])
            avg_precision = np.mean(self.kfold_metrics[model_name]['precision'])
            avg_recall = np.mean(self.kfold_metrics[model_name]['recall'])
            avg_f1 = np.mean(self.kfold_metrics[model_name]['f1'])
            avg_mcc = np.mean(self.kfold_metrics[model_name]['mcc'])
            avg_pcc = np.mean(self.kfold_metrics[model_name]['pcc'])
            
            std_accuracy = np.std(self.kfold_metrics[model_name]['accuracy'])
            std_precision = np.std(self.kfold_metrics[model_name]['precision'])
            std_recall = np.std(self.kfold_metrics[model_name]['recall'])
            std_f1 = np.std(self.kfold_metrics[model_name]['f1'])
            std_mcc = np.std(self.kfold_metrics[model_name]['mcc'])
            std_pcc = np.std(self.kfold_metrics[model_name]['pcc'])
            
            print(f"\n===== {CONFIG['k_folds']}-Fold CV Results for {model_name} =====")
            print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
            print(f"Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
            print(f"Average MCC: {avg_mcc:.4f} ± {std_mcc:.4f}")
            print(f"Average PCC: {avg_pcc:.4f} ± {std_pcc:.4f}")
            
            # Store the average metrics in the main metrics dictionary
            self.metrics[f"{model_name}_CV"] = {
                'accuracy': avg_accuracy,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'mcc': avg_mcc,
                'pcc': avg_pcc
            }
            
            # Add to performance table
            self.model_performances.append([
                f"{model_name} (CV)",
                f"{avg_accuracy:.4f} ± {std_accuracy:.4f}",
                f"{avg_precision:.4f} ± {std_precision:.4f}",
                f"{avg_recall:.4f} ± {std_recall:.4f}",
                f"{avg_f1:.4f} ± {std_f1:.4f}",
                f"{avg_mcc:.4f} ± {std_mcc:.4f}",
                f"{avg_pcc:.4f} ± {std_pcc:.4f}",
                f"{np.mean([v for k,v in self.training_times.items() if k.startswith(f'{model_name}_fold')]):.2f}s"
            ])

    def evaluate_model(self, model, test_loader, model_name, fold=None):
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []  # To store probabilities for PCC
        
        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                
                if isinstance(model, HybridAutoencoderClassifier):
                    _, outputs = model(signals)
                else:
                    outputs = model(signals)
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        # For PCC, we'll use the probability of the positive class (assuming binary classification)
        if CONFIG['num_classes'] == 2:
            positive_probs = [p[1] for p in all_probs]
            pcc, _ = pearsonr(positive_probs, all_labels)
        else:
            # For multiclass, we'll use one-vs-rest approach for PCC
            pcc = 0.0
            for class_idx in range(CONFIG['num_classes']):
                class_labels = [1 if x == class_idx else 0 for x in all_labels]
                class_probs = [p[class_idx] for p in all_probs]
                class_pcc, _ = pearsonr(class_probs, class_labels)
                pcc += class_pcc
            pcc /= CONFIG['num_classes']
        
        # Store metrics
        if fold is not None:
            if model_name not in self.kfold_metrics:
                self.kfold_metrics[model_name] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'mcc': [],
                    'pcc': []
                }
            self.kfold_metrics[model_name]['accuracy'].append(accuracy)
            self.kfold_metrics[model_name]['precision'].append(precision)
            self.kfold_metrics[model_name]['recall'].append(recall)
            self.kfold_metrics[model_name]['f1'].append(f1)
            self.kfold_metrics[model_name]['mcc'].append(mcc)
            self.kfold_metrics[model_name]['pcc'].append(pcc)
            
            # Print fold results
            print(f"\nFold {fold} Results for {model_name}:")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"MCC: {mcc:.4f}, PCC: {pcc:.4f}")
        else:
            self.metrics[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc,
                'pcc': pcc
            }
        
        # Add to performance table (only for final evaluation)
        if fold is None:
            self.model_performances.append([
                model_name,
                f"{accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{mcc:.4f}",
                f"{pcc:.4f}",
                f"{self.training_times.get(model_name, 0):.2f}s"
            ])
            
            # Classification report
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(
                all_labels, 
                all_preds, 
                target_names=DISTURBANCE_TYPES.values(),
                digits=4,
                zero_division=0
            ))
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(4, 3))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                annot_kws={"size": 12},
                xticklabels=DISTURBANCE_TYPES.values(),
                yticklabels=DISTURBANCE_TYPES.values(),
                cbar_kws={'label': 'Count'}
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks()
            plt.yticks()
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{model_name}.png')
            plt.show()
        
        return all_labels, all_preds
    
    def plot_combined_metrics(self):
        """Plot comparison of all models' metrics"""
        if not self.metrics:
            print("No metrics to plot. Train and evaluate models first.")
            return
        
        models = list(self.metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'pcc']
        
        values = {
            'Accuracy': [self.metrics[m]['accuracy'] for m in models],
            'Precision': [self.metrics[m]['precision'] for m in models],
            'Recall': [self.metrics[m]['recall'] for m in models],
            'F1-Score': [self.metrics[m]['f1'] for m in models],
            'MCC': [self.metrics[m]['mcc'] for m in models],
            'PCC': [self.metrics[m]['pcc'] for m in models]
        }
        
        x = np.arange(len(models))
        width = 0.15
        multiplier = 0
        
        fig, ax = plt.subplots(figsize=(18, 8))
        
        for attribute, measurement in values.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3, fmt='%.3f')
            multiplier += 1
        
        ax.set_title('Model Performance Comparison', fontsize=20)
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(models, fontsize=14, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()
    
    def plot_training_curves(self, history_dict):
        """Plot training curves for all models"""
        plt.figure(figsize=(15, 10))
        
        # Plot training loss
        plt.subplot(2, 2, 1)
        for model_name, (train_loss, val_loss, val_acc) in history_dict.items():
            plt.plot(train_loss, label=f'{model_name} Train')
        plt.title('Training Loss Comparison', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Plot validation loss
        plt.subplot(2, 2, 2)
        for model_name, (train_loss, val_loss, val_acc) in history_dict.items():
            plt.plot(val_loss, label=f'{model_name} Val')
        plt.title('Validation Loss Comparison', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Plot validation accuracy
        plt.subplot(2, 2, 3)
        for model_name, (train_loss, val_loss, val_acc) in history_dict.items():
            plt.plot(val_acc, label=f'{model_name}')
        plt.title('Validation Accuracy Comparison', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves_comparison.png')
        plt.show()
    
    def show_performance_table(self):
        """Display performance metrics table"""
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "MCC", "PCC", "Training Time"]
        print("\nModel Performance Comparison:")
        print(tabulate(self.model_performances, headers=headers, tablefmt="grid"))

# ================== VISUALIZATION ================== #
def plot_signals_with_fft(signals, labels):
    """Plot time-domain signal and FFT side-by-side for one example of each class"""
    signals_np = signals.squeeze().numpy() if torch.is_tensor(signals) else signals.squeeze()
    labels_np = labels.numpy() if torch.is_tensor(labels) else labels
    
    present_classes = np.unique(labels_np)
    n_classes = len(present_classes)
    
    fig, axs = plt.subplots(n_classes, 2, figsize=(15, 3 * n_classes))
    if n_classes == 1:
        axs = axs.reshape(1, -1)
    
    plotted_classes = set()
    
    for i, (signal, label) in enumerate(zip(signals_np, labels_np)):
        if label not in plotted_classes and label in DISTURBANCE_TYPES:
            row_idx = len(plotted_classes)
            
            axs[row_idx, 0].plot(signal)
            axs[row_idx, 0].set_title(f'Class {label}: {DISTURBANCE_TYPES[label]} (Time Domain)', fontsize=20)
            axs[row_idx, 0].set_ylabel('Amplitude', fontsize=20)
            axs[row_idx, 0].set_xlabel('Samples', fontsize=20)
            axs[row_idx, 0].grid(True)
            
            n = len(signal)
            yf = fft(signal)
            xf = np.linspace(0, 1.0/(2.0), n//2)
            axs[row_idx, 1].plot(xf, 2.0/n * np.abs(yf[:n//2]))
            axs[row_idx, 1].set_title(f'Class {label}: {DISTURBANCE_TYPES[label]} (Frequency Domain)', fontsize=20)
            axs[row_idx, 1].set_ylabel('Magnitude', fontsize=20)
            axs[row_idx, 1].set_xlabel('Frequency', fontsize=20)
            axs[row_idx, 1].grid(True)
            
            plotted_classes.add(label)
            
            if len(plotted_classes) == n_classes:
                break
    
    plt.tight_layout()
    plt.savefig('signal_examples.png')
    plt.show()

def plot_class_distribution(labels):
    """Plot the distribution of classes"""
    if torch.is_tensor(labels):
        labels = labels.numpy()
    
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(15, 5))
    plt.bar([DISTURBANCE_TYPES[u] for u in unique], counts)
    plt.title('Class Distribution', fontsize=20)
    plt.xlabel('Disturbance Type', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

def plot_signal_statistics(signals, labels):
    """Plot statistical features of signals per class"""
    if torch.is_tensor(signals):
        signals = signals.squeeze().numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    means = np.mean(signals, axis=1)
    stds = np.std(signals, axis=1)
    maxs = np.max(signals, axis=1)
    mins = np.min(signals, axis=1)
    
    for i, (stat, name) in enumerate(zip(
        [means, stds, maxs, mins],
        ['Mean', 'Standard Deviation', 'Maximum', 'Minimum']
    )):
        ax = axes[i//2, i%2]
        for label in np.unique(labels):
            ax.hist(stat[labels == label], alpha=0.5, label=DISTURBANCE_TYPES[label])
        ax.set_title(f'{name} Distribution by Class', fontsize=26)
        ax.set_xlabel(name, fontsize=26)
        ax.set_ylabel('Frequency', fontsize=26)
        ax.legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig('signal_statistics.png')
    plt.show()

def plot_latent_space(model, data_loader, model_name):
    """Visualize the latent space using t-SNE"""
    model.eval()
    all_features = []  # Changed from 'features' to 'all_features'
    labels_list = []
    
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(CONFIG['device'])
            
            if isinstance(model, HybridAutoencoderClassifier):
                # For autoencoder-classifier hybrid
                encoded = model.encoder(signals)
                x = encoded.view(encoded.size(0), -1).cpu().numpy()
            elif isinstance(model, ResNet1D):
                # For ResNet1D
                x = model.conv1(signals)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.avgpool(x)
                x = x.view(x.size(0), -1).cpu().numpy()
            elif isinstance(model, InceptionTime):
                # For InceptionTime
                x = model.inception1(signals)
                res = model.residual1(signals)
                x = x + res
                x = nn.functional.relu(x)
                
                x = model.inception2(x)
                res = model.residual2(x)
                x = x + res
                x = nn.functional.relu(x)
                
                x = model.inception3(x)
                x = model.avgpool(x)
                x = x.view(x.size(0), -1).cpu().numpy()
            elif isinstance(model, (TransformerClassifier, TimeSeriesTransformer)):
                # For Transformer models
                if isinstance(model, TransformerClassifier):
                    x = model.embedding(signals)
                    x = model.bn_embed(x)
                    x = nn.functional.relu(x)
                    x = x.permute(0, 2, 1)
                    x = model.pos_encoder(x)
                    x = model.transformer(x)
                    x = x.permute(0, 2, 1)
                    x = model.avgpool(x).squeeze(-1)
                else:  # TimeSeriesTransformer
                    x = model.patch_embed(signals)
                    x = x.permute(2, 0, 1)
                    x = model.position_embedding(x)
                    x = model.transformer_encoder(x)
                    x = x.mean(dim=0)  # Average over sequence length
                    x = model.norm(x)
                x = x.cpu().numpy()
            elif isinstance(model, XCM):
                # For XCM model
                t = F.relu(model.temporal_bn1(model.temporal_conv1(signals)))
                t = F.relu(model.temporal_bn2(model.temporal_conv2(t)))
                t = F.relu(model.temporal_bn3(model.temporal_conv3(t)))
                
                f = F.relu(model.feature_bn1(model.feature_conv1(signals)))
                f = F.relu(model.feature_bn2(model.feature_conv2(f)))
                f = F.relu(model.feature_bn3(model.feature_conv3(f)))
                
                x = torch.cat([t, f], dim=1)
                x = x.mean(dim=2)  # Global average pooling
                x = x.cpu().numpy()
            elif isinstance(model, ConvLSTM):
                # For ConvLSTM model
                x = F.relu(model.bn1(model.conv1(signals)))
                x = F.relu(model.bn2(model.conv2(x)))
                x = F.relu(model.bn3(model.conv3(x)))
                x = x.permute(0, 2, 1)
                lstm_out, _ = model.lstm(x)
                attention_weights = model.attention(lstm_out)
                x = torch.sum(attention_weights * lstm_out, dim=1)
                x = x.cpu().numpy()
            elif isinstance(model, DenseNet1D):
                # For DenseNet1D
                features = model.features(signals)
                x = nn.functional.relu(features, inplace=True)
                x = nn.functional.adaptive_avg_pool1d(x, 1)
                x = x.view(features.size(0), -1).cpu().numpy()
            else:  # Default case (CNN)
                # For standard CNN
                x = model.features(signals)
                x = x.view(x.size(0), -1).cpu().numpy()
            
            all_features.append(x)  # Changed from 'features' to 'all_features'
            labels_list.append(labels.cpu().numpy())
    
    all_features = np.concatenate(all_features)  # Changed from 'features' to 'all_features'
    labels_list = np.concatenate(labels_list)
    
    # First reduce dimensionality with PCA
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(all_features)  # Changed from 'features' to 'all_features'
    
    # Then apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)
    
    plt.figure(figsize=(4, 3))
    for label in np.unique(labels_list):
        plt.scatter(
            features_tsne[labels_list == label, 0],
            features_tsne[labels_list == label, 1],
            label=DISTURBANCE_TYPES[label],
            alpha=0.6
        )
    plt.title(f't-SNE Visualization - {model_name}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'latent_space_{model_name}.png')
    plt.show()

# ================== MAIN EXECUTION ================== #
def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    signals_tensor, labels_tensor = load_and_preprocess_data()
    
    # Plot data visualizations
    print("\nVisualizing data characteristics...")
    plot_signals_with_fft(signals_tensor, labels_tensor)
    plot_class_distribution(labels_tensor)
    plot_signal_statistics(signals_tensor, labels_tensor)
    
    # Create full dataset
    dataset = TensorDataset(signals_tensor, labels_tensor)
    
    # Initialize models and trainer
    models = {
        'CNN': CNN(),
        'HybridAE': HybridAutoencoderClassifier(),
        'ResNet1D': ResNet1D(),
        'InceptionTime': InceptionTime(),
        'Transformer': TransformerClassifier(),
        'XCM': XCM(),
        'ConvLSTM': ConvLSTM(),
        'DenseNet1D': DenseNet1D()
    }
    
    trainer = ModelTrainer()
    history_dict = {}
    
    # Perform k-fold cross validation for each model
    for name, model in models.items():
        trainer.cross_validate(model, dataset, name)
    
    # Now train each model on the full training set and evaluate on the test set
    # Split into train, validation, test (60-20-20)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Train and evaluate each model on the full training set
    for name, model in models.items():
        print(f"\n===== Training {name} on Full Training Set =====")
        train_loss, val_loss, val_acc = trainer.train_model(model, train_loader, val_loader, name)
        history_dict[name] = (train_loss, val_loss, val_acc)
        
        # Plot training history
        plt.figure(figsize=(4, 3))
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title(f'{name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(val_acc)
        plt.title(f'{name} Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{name}.png')
        plt.show()
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(f'best_{name}.pth'))
        print(f"\nEvaluating {name} on test set...")
        trainer.evaluate_model(model, test_loader, name)
        
        # Visualize latent space
        print(f"\nVisualizing latent space for {name}...")
        plot_latent_space(model, test_loader, name)
    
    # Plot combined metrics and training curves
    trainer.plot_combined_metrics()
    trainer.plot_training_curves(history_dict)
    trainer.show_performance_table()

if __name__ == "__main__":
    main()