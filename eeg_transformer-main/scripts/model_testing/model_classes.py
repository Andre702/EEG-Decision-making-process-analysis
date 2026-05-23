import torch
import torch.nn as nn

# MODEL -----------------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, model_class_name='TemporalTransformer', method='raw'):
        super(FeatureExtractor, self).__init__()
        self.model_class_name = model_class_name
        self.method = method

    def forward(self, x):
        if self.method == 'raw':
            if x.ndim == 3:
                x = x.unsqueeze(1)
            return x
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :, :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.3)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_classes=2, feature_method='raw', feature_extractor=None):
        super(TemporalTransformer, self).__init__()
        self.feature_method = feature_method

        # Feature Extractor (for now only raw tested) -----------------
        if feature_method in ['raw', 'cnn', 'stft']:
            self.feature_extractor = FeatureExtractor(
                model_class_name=self.__class__.__name__,
                method=feature_method
            )
        elif feature_method == 'csp':
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = None

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer blocks
        self.transformer = nn.Sequential(
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead),
            TransformerBlock(d_model, nhead)
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, channels, time)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)

        # (time = sequence dimension)
        if self.feature_method == 'raw':
            x = x.squeeze(1).permute(2, 0, 1)

        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)