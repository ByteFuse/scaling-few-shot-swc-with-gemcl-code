import math
import torch
import torch.nn as nn
from einops import rearrange

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return x


class FeatureEncoder(nn.Module):
    """Feature encoder that processes MFCC features."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        layers = []
        
        # First layer
        layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=10, stride=5, padding=4))
        layers.append(nn.GroupNorm(32, hidden_dim))
        layers.append(nn.GELU())
        
        # Additional layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(32, hidden_dim))
            layers.append(nn.GELU())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim, seq_len)
        Returns:
            Tensor of shape (batch_size, hidden_dim, reduced_seq_len)
        """
        return self.layers(x)


class Wav2Vec2Model(nn.Module):
    """Wav2Vec2-inspired model architecture."""
    
    def __init__(
        self,
        input_dim=13,  # MFCC features
        hidden_dim=768,
        num_transformer_layers=12,
        num_attention_heads=12,
        dropout=0.1,
        num_encoder_layers=3,
        use_feature_encoder=True
    ):
        super().__init__()
        
        self.use_feature_encoder = use_feature_encoder
        
        # Feature encoder (processes raw MFCC features) - optional
        if self.use_feature_encoder:
            self.feature_encoder = FeatureEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_encoder_layers
            )
        else:
            # Linear projection to match hidden_dim if not using conv encoder
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim, seq_len)
            attention_mask: Optional mask for attention
        Returns:
            Tensor of shape (batch_size, seq_len_reduced, hidden_dim)
        """
        if self.use_feature_encoder:
            # Feature encoding with conv layers
            features = self.feature_encoder(x)  # (batch, hidden_dim, seq_len_reduced)
            # Transpose for transformer: (seq_len, batch, hidden_dim)
            features = rearrange(features, 'batch hidden_dim seq_len -> seq_len batch hidden_dim')
        else:
            # Direct projection without conv layers
            # x shape: (batch, input_dim, seq_len)
            x = rearrange(x, 'batch input_dim seq_len -> batch seq_len input_dim')
            features = self.input_projection(x)  # (batch, seq_len, hidden_dim)
            features = rearrange(features, 'batch seq_len hidden_dim -> seq_len batch hidden_dim')

        
        # Add positional encoding
        features = self.pos_encoder(features)
        features = self.dropout(features)
        
        # Transformer encoding
        encoded = self.transformer_encoder(features, src_key_padding_mask=attention_mask)
        encoded = self.layer_norm(encoded)
        
        # Back to (batch, seq_len, hidden_dim)
        encoded = rearrange(encoded, 'seq_len batch hidden_dim -> batch seq_len hidden_dim')
        
        return encoded