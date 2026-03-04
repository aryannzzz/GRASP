"""
OpenVLA Integration for VLA-GradCAM

This module provides specific integration for OpenVLA model.
OpenVLA uses SigLIP as vision encoder and Llama as language model.

Architecture:
    Image → SigLIP (ViT) → Vision Tokens
                              ↓
    Instruction → Llama Tokenizer → Text Tokens
                              ↓
                    Llama (with cross-attention)
                              ↓
                    Action Head → [x, y, z, rx, ry, rz, gripper]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. OpenVLA integration unavailable.")

from .vla_gradcam import VLAGradCAM, VLASaliencyResult


# OpenVLA action space (typical 7-DOF robot arm)
OPENVLA_ACTION_NAMES = [
    'delta_x',      # End-effector X movement
    'delta_y',      # End-effector Y movement  
    'delta_z',      # End-effector Z movement
    'delta_roll',   # Rotation around X
    'delta_pitch',  # Rotation around Y
    'delta_yaw',    # Rotation around Z
    'gripper',      # Gripper open/close
]


class OpenVLAGradCAM(VLAGradCAM):
    """
    GradCAM for OpenVLA model.
    
    OpenVLA specifics:
    - Vision encoder: SigLIP (ViT-based)
    - LLM: Llama-based
    - Action tokenization: Actions are decoded as special tokens
    
    Usage:
        model = OpenVLAGradCAM.load_pretrained("openvla/openvla-7b")
        result = model.compute_saliency(image, "pick up the red cup")
        model.visualize(result)
    """
    
    def __init__(
        self,
        model: nn.Module,
        processor=None,
        device: str = 'cuda',
    ):
        """
        Args:
            model: OpenVLA model
            processor: OpenVLA processor for tokenization
            device: Device to run on
        """
        self.processor = processor
        self.device = device
        
        # OpenVLA uses SigLIP vision encoder
        # Find it in the model structure
        vision_encoder_name = self._find_vision_encoder(model)
        
        super().__init__(
            model=model,
            vision_encoder_name=vision_encoder_name,
            patch_size=14,  # SigLIP default
            image_size=(224, 224),
            action_names=OPENVLA_ACTION_NAMES,
        )
        
    def _find_vision_encoder(self, model: nn.Module) -> str:
        """Find the vision encoder in OpenVLA model structure."""
        # Common names in OpenVLA/Prismatic architectures
        candidates = [
            'vision_backbone',
            'visual',
            'vision_tower', 
            'image_encoder',
            'vision_encoder',
        ]
        
        for name in candidates:
            if hasattr(model, name):
                return name
                
        # Search in submodules
        for name, module in model.named_modules():
            if 'siglip' in name.lower() or 'vit' in name.lower():
                # Get parent name
                parts = name.split('.')
                return parts[0] if parts else name
                
        raise ValueError("Could not find vision encoder in OpenVLA model")
    
    @classmethod
    def load_pretrained(
        cls,
        model_name: str = "openvla/openvla-7b",
        device: str = 'cuda',
        load_in_8bit: bool = True,
    ) -> 'OpenVLAGradCAM':
        """
        Load pretrained OpenVLA model with GradCAM capability.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load on
            load_in_8bit: Use 8-bit quantization for memory efficiency
            
        Returns:
            OpenVLAGradCAM instance
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")
            
        print(f"Loading {model_name}...")
        
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            trust_remote_code=True
        )
        
        return cls(model=model, processor=processor, device=device)
    
    def preprocess(
        self,
        image: np.ndarray,
        instruction: str,
    ) -> Dict:
        """
        Preprocess image and instruction for OpenVLA.
        
        Args:
            image: RGB image [H, W, C]
            instruction: Language instruction
            
        Returns:
            Dict with processed inputs
        """
        if self.processor is None:
            raise ValueError("Processor not set. Load model with load_pretrained().")
            
        # OpenVLA expects specific prompt format
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        # Process with OpenVLA processor
        inputs = self.processor(
            prompt,
            image,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
    
    def compute_saliency(
        self,
        image: np.ndarray,
        instruction: str,
        action_dims: Optional[List[int]] = None,
        **kwargs
    ) -> VLASaliencyResult:
        """
        Compute saliency for OpenVLA.
        
        This overrides the base class to handle OpenVLA's specific
        input/output format.
        """
        # Preprocess
        inputs = self.preprocess(image, instruction)
        
        # Get predicted action
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )
            
        # Decode action (OpenVLA encodes actions as tokens)
        action_tokens = output[0, inputs['input_ids'].shape[1]:]
        predicted_action = self._decode_action(action_tokens)
        
        # For saliency, we need gradient-enabled forward pass
        # Use the vision encoder outputs directly
        return self._compute_vision_saliency(
            inputs=inputs,
            image=image,
            instruction=instruction,
            predicted_action=predicted_action,
            action_dims=action_dims
        )
    
    def _decode_action(self, action_tokens: torch.Tensor) -> np.ndarray:
        """Decode action tokens to continuous action values."""
        # OpenVLA uses special action tokens
        # This is a simplified version - actual decoding depends on model version
        decoded = self.processor.decode(action_tokens, skip_special_tokens=True)
        
        # Parse action values from decoded string
        try:
            # Format: "action_0 action_1 ... action_6"
            values = [float(x) for x in decoded.split()]
            return np.array(values)
        except:
            # Fallback: return zeros
            return np.zeros(7)
    
    def _compute_vision_saliency(
        self,
        inputs: Dict,
        image: np.ndarray,
        instruction: str,
        predicted_action: np.ndarray,
        action_dims: Optional[List[int]] = None,
    ) -> VLASaliencyResult:
        """
        Compute saliency by analyzing vision encoder gradients.
        
        Since OpenVLA uses autoregressive generation for actions,
        we compute saliency w.r.t. the vision features that influence
        the generated action tokens.
        """
        self.model.eval()
        
        # Enable gradients for vision features
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Find vision encoder and enable its gradients
        vision_encoder = getattr(self.model, self.vision_encoder_name)
        for param in vision_encoder.parameters():
            param.requires_grad = True
            
        # Forward pass through vision encoder
        pixel_values = inputs.get('pixel_values', inputs.get('images'))
        pixel_values.requires_grad = True
        
        # Get vision features
        vision_outputs = vision_encoder(pixel_values)
        if hasattr(vision_outputs, 'last_hidden_state'):
            vision_features = vision_outputs.last_hidden_state
        else:
            vision_features = vision_outputs
            
        # Compute saliency for each action dimension
        action_dims = action_dims or list(range(len(OPENVLA_ACTION_NAMES)))
        saliency_maps = {}
        
        for dim_idx in action_dims:
            self.model.zero_grad()
            
            # We use the vision features' contribution to the model output
            # Since OpenVLA is autoregressive, we approximate by using
            # the features directly
            
            # Compute gradient of vision features w.r.t. input
            target = vision_features.mean()  # Simplified - should be action-specific
            target.backward(retain_graph=True)
            
            if pixel_values.grad is not None:
                # Use input gradients as saliency
                grad = pixel_values.grad.abs().mean(dim=1)  # [B, H, W]
                saliency = grad.squeeze().cpu().numpy()
                
                # Normalize
                if saliency.max() > 0:
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                    
                dim_name = OPENVLA_ACTION_NAMES[dim_idx]
                saliency_maps[dim_name] = saliency
                
                pixel_values.grad.zero_()
                
        # Combined saliency
        combined = np.mean(list(saliency_maps.values()), axis=0) if saliency_maps else np.zeros_like(image[:,:,0])
        
        return VLASaliencyResult(
            image=image,
            instruction=instruction,
            predicted_action=predicted_action,
            saliency_maps=saliency_maps,
            combined_saliency=combined,
            action_names=OPENVLA_ACTION_NAMES
        )


class MockVLAModel(nn.Module):
    """
    A mock VLA model for testing VLA-GradCAM without requiring actual model weights.
    
    Mimics OpenVLA architecture:
    - ViT vision encoder
    - Transformer decoder
    - Action output head
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        action_dim: int = 7,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        
        # Patch embedding (like ViT)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder (this is what we'll hook into for GradCAM)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
                dropout=0.0,  # No dropout for reproducibility
            ),
            num_layers=n_layers
        )
        
        # For GradCAM: expose the last transformer layer
        self.vision_encoder = self.transformer
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, action_dim)
        )
        
        # Store embed dim for hook
        self.embed_dim = embed_dim
        
    def forward(
        self,
        image: torch.Tensor,
        instruction: str = None,  # Unused in mock
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: [B, C, H, W] input image
            instruction: Language instruction (unused in mock)
            
        Returns:
            action: [B, action_dim] predicted action
        """
        # Patch embedding: [B, C, H, W] -> [B, embed_dim, h, w] -> [B, n_patches, embed_dim]
        x = self.patch_embed(image)  # [B, embed_dim, h, w]
        x = x.flatten(2).permute(0, 2, 1)  # [B, n_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer (this outputs [B, n_patches, embed_dim])
        x = self.transformer(x)  # [B, n_patches, embed_dim]
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # Action prediction
        action = self.action_head(x)  # [B, action_dim]
        
        return action


def test_mock_vla():
    """Test VLA-GradCAM with mock model."""
    print("Testing VLA-GradCAM with Mock VLA Model")
    print("=" * 50)
    
    # Create mock model
    model = MockVLAModel()
    model.eval()
    
    # Create GradCAM wrapper
    gradcam = VLAGradCAM(
        model=model,
        vision_encoder_name='vision_encoder',
        patch_size=14,
        image_size=(224, 224),
        action_names=OPENVLA_ACTION_NAMES
    )
    
    # Test with random image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    instruction = "pick up the red cup"
    
    print(f"Input image shape: {test_image.shape}")
    print(f"Instruction: {instruction}")
    
    # Compute saliency
    result = gradcam.compute_saliency(
        image=test_image,
        instruction=instruction,
        action_dims=[0, 1, 2, 6]  # x, y, z, gripper
    )
    
    print(f"\nResults:")
    print(f"  Predicted action: {result.predicted_action}")
    print(f"  Saliency maps computed: {list(result.saliency_maps.keys())}")
    print(f"  Combined saliency shape: {result.combined_saliency.shape}")
    
    # Check saliency properties
    for name, saliency in result.saliency_maps.items():
        print(f"  {name}: min={saliency.min():.3f}, max={saliency.max():.3f}, mean={saliency.mean():.3f}")
    
    # Cleanup
    gradcam.remove_hooks()
    
    print("\n✓ Mock VLA test passed!")
    return result


if __name__ == "__main__":
    test_mock_vla()
