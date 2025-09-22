import torch
from src.fiber_fusion import FIBERCrossModalFusion

# Config dims from configs/bitmar_coco.yaml
vision_dim = 64
text_dim = 64
hidden_dim = 64
num_heads = 2
num_layers = 1

# Instantiate module
fusion = FIBERCrossModalFusion(
    vision_dim=vision_dim,
    text_dim=text_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=0.1,
    use_relative_pos=True,
)

# Create dummy inputs
B = 2
seq_len = 8
vision_features = torch.randn(B, vision_dim)
text_features = torch.randn(B, seq_len, text_dim)
text_mask = torch.ones(B, seq_len, dtype=torch.long)

# Run forward
with torch.no_grad():
    out = fusion(vision_features, text_features, text_mask)

print('OK: forward returned keys:', list(out.keys()))
print('vision_features shape:', out['vision_features'].shape)
print('text_features shape:', out['text_features'].shape)
print('similarity shape:', out['cross_modal_similarity'].shape)

