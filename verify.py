"""Quick verification: test model forward pass with dummy data (no dataset needed)."""
import torch
import easydict

print("[*] Setting up model...")

args = easydict.EasyDict()
args.vision_model = 'dinov2_vitb14'
args.llm_model = 'Qwen/Qwen3-0.6B'
args.num_id_tokens = 4
args.num_vpt_tokens = 2
args.num_icl_samples = 8
args.num_icl_bs = 2
args.icl_loss_weight = 1.0
args.ot_loss_weight = 0.01

from models import Model
print("[*] Building model (this downloads DINOv2 and Qwen3 on first run)...")
model = Model(args).cuda() if torch.cuda.is_available() else Model(args)
model.train()

print("[*] Running forward pass with dummy data...")
image_crops = torch.randn(2, 2, 3, 224, 224)
if torch.cuda.is_available():
    image_crops = image_crops.cuda()
labels = torch.randint(0, 10, (2,))
if torch.cuda.is_available():
    labels = labels.cuda()

outputs = model(image_crops, labels)
print()
print("[+] Forward pass successful!")
print(f"    loss:      {outputs['loss'].item():.4f}")
print(f"    id_loss:   {outputs['id_loss'].item():.4f}")
print(f"    icl_loss:  {outputs['icl_loss'].item():.4f}")
print(f"    ot_loss:   {outputs['ot_loss'].item():.4f}")
print(f"    std:       {outputs['std'].item():.4f}")
print(f"    features shape: {outputs['features'].shape}")
print(f"    prompts shape:  {outputs['prompts'].shape}")
print()
print("[+] VICP model is working correctly!")
print("[*] Next: download ShopID10K dataset and run: python run.py --preset quick")
