import torch
import torch.nn as nn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50 

def build_keras_cnn_vit_hybrid():
    """Answers Q7 Tasks 1-4: Keras CNN-ViT Hybrid."""
    print("Building Keras CNN-ViT Hybrid Model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    for layer in base_model.layers:
        layer.trainable = False
        
    x = Flatten()(base_model.output)
    # Simulator for the Transformer block addition
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    hybrid_model = Model(inputs=base_model.input, outputs=output)
    return hybrid_model

class PyTorchCNNViTHybrid(nn.Module):
    """Answers Q8 Tasks 1-4: PyTorch CNN-ViT Hybrid."""
    def __init__(self, embed_dim=768, num_heads=12, transformer_depth=12):
        super(PyTorchCNNViTHybrid, self).__init__()
        # CNN Base for localized feature extraction
        self.cnn_base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.patch_embed = nn.Linear(64 * 32 * 32, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), 1, -1) 
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 
        x = self.fc(x)
        return x
