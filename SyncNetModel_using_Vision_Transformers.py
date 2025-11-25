# SyncNetModel_using_Vision_Transformers.py

#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

def save(model, filename):
    with open(filename, "wb") as f:
        torch.save(model, f)
        print("%s saved."%filename)

def load(filename):
    net = torch.load(filename)
    return net

class PatchEmbedding2D(nn.Module):
    """
    Splits 2D Audio Spectrogram into patches and projects them.
    Input: (B, C, H, W) -> Output: (B, N_patches, Embed_Dim)
    """
    def __init__(self, in_channels=1, patch_size=16, embed_dim=256, img_size=(256, 256)):
        super().__init__()
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        # We use a Conv2d with stride=patch_size to implement the linear projection of patches efficiently
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, Embed_Dim, Grid_H, Grid_W)
        x = x.flatten(2)  # (B, Embed_Dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, Embed_Dim)
        return x

class PatchEmbedding3D(nn.Module):
    """
    Splits 3D Video Volume (Lip frames) into 'Tubelets' (3D patches).
    Input: (B, C, T, H, W) -> Output: (B, N_patches, Embed_Dim)
    """
    def __init__(self, in_channels=3, patch_size=(1, 16, 16), embed_dim=256, vid_size=(5, 96, 96)):
        super().__init__()
        # patch_size format: (Time_Patch, Height_Patch, Width_Patch)
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.grid_t = vid_size[0] // patch_size[0]
        self.grid_h = vid_size[1] // patch_size[1]
        self.grid_w = vid_size[2] // patch_size[2]
        self.num_patches = self.grid_t * self.grid_h * self.grid_w

        # Use Conv3d to project 3D cubes to vectors
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, Embed_Dim, Grid_T, Grid_H, Grid_W)
        x = x.flatten(2)  # (B, Embed_Dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, Embed_Dim)
        return x

class TransformerEncoder(nn.Module):
    """
    Standard Transformer Encoder Block
    """
    def __init__(self, embed_dim, num_heads, layers, mlp_ratio=4.0):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True # Pre-Norm usually converges faster
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, N_patches + 1, Embed_Dim)
        x = self.transformer_encoder(x)
        return self.norm(x)

class SyncTransformer(nn.Module):
    def __init__(self, num_layers_in_fc_layers=1024, 
                 aud_img_size=(128, 256),  # Expected spectrogram size
                 lip_img_size=(5, 96, 96), # Expected lip video size (T, H, W)
                 embed_dim=256, 
                 depth=4, 
                 num_heads=4):
        super(SyncTransformer, self).__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32

        # --- AUDIO STREAM (Spectrogram -> 2D Patches -> Transformer) ---
        self.aud_patch_embed = PatchEmbedding2D(
            in_channels=1, 
            patch_size=16, 
            embed_dim=embed_dim, 
            img_size=aud_img_size
        )
        # Learnable Position Embedding for Audio
        self.aud_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.aud_patch_embed.num_patches, embed_dim))
        self.aud_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.aud_transformer = TransformerEncoder(embed_dim, num_heads, depth)
        
        # Projection Head to match your requested 1024 output
        self.netfcaud = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        # --- LIP STREAM (Video 3D -> Tubelets -> Transformer) ---
        self.lip_patch_embed = PatchEmbedding3D(
            in_channels=3, 
            patch_size=(1, 16, 16), # Capture 1 frame deep, 16x16 spatial
            embed_dim=embed_dim, 
            vid_size=lip_img_size
        )
        # Learnable Position Embedding for Lips
        self.lip_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.lip_patch_embed.num_patches, embed_dim))
        self.lip_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.lip_transformer = TransformerEncoder(embed_dim, num_heads, depth)

        # Projection Head
        self.netfclip = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize patch embeddings and normalization
        for name, p in self.named_parameters():
            if 'pos_embed' in name or 'cls_token' in name:
                nn.init.trunc_normal_(p, std=0.02)
            elif 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_aud(self, x):
        """
        Input x: (Batch, 1, H, W) - Spectrogram
        """
        # 1. Patchify
        x = self.aud_patch_embed(x) # (B, N, E)
        
        # 2. Add CLS Token
        cls_token = self.aud_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) # (B, N+1, E)

        # 3. Add Position Embeddings
        x = x + self.aud_pos_embed

        # 4. Transformer
        x = self.aud_transformer(x)

        # 5. Extract CLS token output (index 0)
        out = x[:, 0]

        # 6. Final Projection
        out = self.netfcaud(out)
        return out

    def forward_lip(self, x):
        """
        Input x: (Batch, 3, T, H, W) - Lip Video
        """
        # 1. Patchify (Tubelets)
        x = self.lip_patch_embed(x) # (B, N, E)

        # 2. Add CLS Token
        cls_token = self.lip_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 3. Add Position Embeddings
        x = x + self.lip_pos_embed

        # 4. Transformer
        x = self.lip_transformer(x)

        # 5. Extract CLS token output
        out = x[:, 0]

        # 6. Final Projection
        out = self.netfclip(out)
        return out

    def forward_lipfeat(self, x):
        """
        Returns the raw features from the transformer before the final projection layer.
        """
        x = self.lip_patch_embed(x)
        cls_token = self.lip_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.lip_pos_embed
        x = self.lip_transformer(x)
        return x[:, 0]

# Example Usage to verify dimensions
if __name__ == "__main__":
    # Create Model
    # Note: Adjust aud_img_size and lip_img_size to match your actual data pipeline dimensions
    model = SyncTransformer(
        num_layers_in_fc_layers=1024,
        aud_img_size=(128, 256),
        lip_img_size=(5, 96, 96) 
    )

    # Dummy Audio Input: (Batch, 1, Freq, Time)
    dummy_audio = torch.randn(2, 1, 128, 256)
    
    # Dummy Lip Input: (Batch, 3, Time, Height, Width)
    dummy_lip = torch.randn(2, 3, 5, 96, 96)

    aud_out = model.forward_aud(dummy_audio)
    lip_out = model.forward_lip(dummy_lip)

    print("Audio Output Shape:", aud_out.shape) # Should be [2, 1024]
    print("Lip Output Shape:", lip_out.shape)   # Should be [2, 1024]
