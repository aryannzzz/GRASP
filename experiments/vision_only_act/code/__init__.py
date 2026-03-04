"""Modified ACT Policy with Images in VAE Encoder.

This is a modified version of the Action Chunking Transformer (ACT) policy
where the VAE encoder takes images as input in addition to joint states and actions.
This helps prevent overfitting on proprioceptive information alone by conditioning
the latent distribution on visual information.

Usage:
    from lerobot.policies.modified_act import ModifiedACTConfig, ModifiedACTPolicy
    
    config = ModifiedACTConfig(
        vae_encoder_use_images=True,  # The key modification
        vae_encoder_image_pooling="global_avg",  # or "spatial"
    )
    policy = ModifiedACTPolicy(config)
"""

from lerobot.policies.modified_act.configuration_modified_act import ModifiedACTConfig
from lerobot.policies.modified_act.modeling_modified_act import ModifiedACT, ModifiedACTPolicy
from lerobot.policies.modified_act.processor_modified_act import make_modified_act_pre_post_processors

__all__ = [
    "ModifiedACTConfig",
    "ModifiedACTPolicy",
    "ModifiedACT",
    "make_modified_act_pre_post_processors",
]
