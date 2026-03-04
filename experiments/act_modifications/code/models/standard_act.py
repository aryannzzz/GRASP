# models/standard_act.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import math

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embeddings"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, length):
        """Generate 1D position embeddings
        Args:
            length: sequence length
        Returns:
            pos_emb: [length, dim]
        """
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32) * 
                            -(math.log(10000.0) / self.dim))
        
        pos_emb = torch.zeros(length, self.dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        
        return pos_emb

class ResNetEncoder(nn.Module):
    """ResNet18 feature extractor for images"""
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True)
        # Remove final FC and avg pool layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Output: [B, 512, H/32, W/32]
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, 512, H/32, W/32]
        """
        return self.features(x)

class StandardACTEncoder(nn.Module):
    """CVAE Encoder - Standard version (no images)"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Latent projection
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, joints, actions):
        """
        Args:
            joints: [B, joint_dim] - current joint state
            actions: [B, T, action_dim] - action chunk
        Returns:
            z_mean: [B, latent_dim]
            z_logvar: [B, latent_dim]
        """
        B, T, _ = actions.shape
        
        # Project inputs
        joint_emb = self.joint_proj(joints).unsqueeze(1)  # [B, 1, hidden_dim]
        action_emb = self.action_proj(actions)  # [B, T, hidden_dim]
        
        # Concatenate
        seq = torch.cat([joint_emb, action_emb], dim=1)  # [B, T+1, hidden_dim]
        
        # Add positional encoding
        pos = self.pos_emb(T + 1).unsqueeze(0).to(seq.device)  # [1, T+1, hidden_dim]
        seq = seq + pos
        
        # Transformer encoding
        encoded = self.transformer(seq)  # [B, T+1, hidden_dim]
        
        # Use last token for latent
        last_token = encoded[:, -1, :]  # [B, hidden_dim]
        
        # Project to latent distribution
        z_mean = self.fc_mean(last_token)
        z_logvar = self.fc_logvar(last_token)
        
        return z_mean, z_logvar

class ACTDecoder(nn.Module):
    """CVAE Decoder / Policy - Shared between standard and modified"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_encoder_layers=4, n_decoder_layers=7, n_heads=8, 
                 feedforward_dim=3200, chunk_size=100, n_cameras=1, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        
        # Image encoder
        self.resnet = ResNetEncoder()
        self.image_proj = nn.Linear(512, hidden_dim)  # ResNet output is 512
        
        # Joint encoder
        self.joint_proj = nn.Linear(joint_dim, hidden_dim)
        
        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encodings
        self.pos_emb_2d = SinusoidalPosEmb(hidden_dim)
        self.pos_emb_1d = SinusoidalPosEmb(hidden_dim)
        
        # Query tokens (learnable)
        self.query_tokens = nn.Embedding(chunk_size, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, images, joints, z):
        """
        Args:
            images: dict with camera_name -> [B, 3, H, W]
            joints: [B, joint_dim]
            z: [B, latent_dim]
        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = joints.shape[0]
        
        # Encode images
        image_features = []
        for cam_name, img in images.items():
            feat = self.resnet(img)  # [B, 512, H', W']
            B, C, H, W = feat.shape
            feat = feat.flatten(2).permute(0, 2, 1)  # [B, H'*W', 512]
            feat = self.image_proj(feat)  # [B, H'*W', hidden_dim]
            
            # Add 2D positional encoding (simplified to 1D for now)
            pos = self.pos_emb_1d(H * W).unsqueeze(0).to(feat.device)
            feat = feat + pos
            
            image_features.append(feat)
        
        # Concatenate all image features
        image_tokens = torch.cat(image_features, dim=1)  # [B, N_tokens, hidden_dim]
        
        # Encode joints
        joint_token = self.joint_proj(joints).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Encode latent
        latent_token = self.latent_proj(z).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Create encoder output (context)
        encoder_output = torch.cat([image_tokens, joint_token, latent_token], dim=1)  # [B, N+2, hidden_dim]
        
        # Create queries
        query_indices = torch.arange(self.chunk_size, device=joints.device)
        queries = self.query_tokens(query_indices).unsqueeze(0).expand(B, -1, -1)  # [B, chunk_size, hidden_dim]
        
        # Add positional encoding to queries
        query_pos = self.pos_emb_1d(self.chunk_size).unsqueeze(0).to(queries.device)
        queries = queries + query_pos
        
        # Decode
        decoded = self.transformer_decoder(queries, encoder_output)  # [B, chunk_size, hidden_dim]
        
        # Project to actions
        actions = self.action_head(decoded)  # [B, chunk_size, action_dim]
        
        return actions

class StandardACT(nn.Module):
    """Complete Standard ACT model (CVAE)"""
    def __init__(self, joint_dim=8, action_dim=8, hidden_dim=512, latent_dim=32,
                 n_encoder_layers=4, n_decoder_layers=7, n_heads=8,
                 feedforward_dim=3200, chunk_size=100, n_cameras=1, dropout=0.1):
        super().__init__()
        
        self.encoder = StandardACTEncoder(
            joint_dim=joint_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.decoder = ACTDecoder(
            joint_dim=joint_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            feedforward_dim=feedforward_dim,
            chunk_size=chunk_size,
            n_cameras=n_cameras,
            dropout=dropout
        )
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, images, joints, actions=None, training=True):
        """
        Args:
            images: dict with camera_name -> [B, 3, H, W]
            joints: [B, joint_dim]
            actions: [B, chunk_size, action_dim] - only needed during training
            training: bool
        Returns:
            If training:
                pred_actions: [B, chunk_size, action_dim]
                z_mean: [B, latent_dim]
                z_logvar: [B, latent_dim]
            Else:
                pred_actions: [B, chunk_size, action_dim]
        """
        if training:
            # Encode to get latent distribution
            z_mean, z_logvar = self.encoder(joints, actions)
            
            # Sample latent
            z = self.reparameterize(z_mean, z_logvar)
        else:
            # During inference, use mean of prior (deterministic)
            # As per ACT paper: "At test time, z is set to zero"
            B = joints.shape[0]
            z = torch.zeros(B, self.latent_dim, device=joints.device)
            z_mean = z_logvar = None
        
        # Decode to get actions
        pred_actions = self.decoder(images, joints, z)
        
        if training:
            return pred_actions, z_mean, z_logvar
        else:
            return pred_actions
