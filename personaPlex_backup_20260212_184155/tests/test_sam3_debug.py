"""Debug: capture webcam, save frame, run SAM3 with multiple prompts, show scores."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np, torch

# Capture frame (wait a few frames for camera warmup)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(30):  # skip first 30 frames for camera warmup
    cap.read()
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: Cannot read from camera"); sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}, mean brightness: {frame.mean():.1f}", flush=True)
cv2.imwrite(os.path.expanduser("~/Downloads/sam3_debug_frame.png"), frame)
print(f"Saved to ~/Downloads/sam3_debug_frame.png", flush=True)

# Load model
from server.config import ServerConfig
config = ServerConfig()
from transformers import Sam3Model, Sam3Processor
processor = Sam3Processor.from_pretrained(config.sam3_model)
model = Sam3Model.from_pretrained(config.sam3_model).to("cuda").eval().half()

# Preprocess
from PIL import Image
pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Run vision
inputs = processor(images=pil_img, text="person", return_tensors="pt")
pixel_values = inputs["pixel_values"].to("cuda").half()
with torch.no_grad():
    vis = model.get_vision_features(pixel_values=pixel_values)
print(f"Vision done. FPN shapes: {[s.shape for s in vis.fpn_hidden_states]}", flush=True)

# Test multiple prompts
prompts = ["person", "face", "chair", "table", "desk", "monitor", "laptop", "wall", "floor"]
for prompt in prompts:
    inp = processor(images=pil_img, text=prompt, return_tensors="pt")
    attn = inp["attention_mask"].to("cuda")
    with torch.no_grad():
        te = model.get_text_features(
            input_ids=inp["input_ids"].to("cuda"),
            attention_mask=attn,
            return_dict=True,
        ).pooler_output.detach()
    # Pad to match
    pad = 32 - attn.shape[1]
    if pad > 0:
        attn = torch.nn.functional.pad(attn, (0, pad), value=0)
        te = torch.nn.functional.pad(te, (0, 0, 0, pad), value=0)

    with torch.no_grad():
        out = model(vision_embeds=vis, text_embeds=te, attention_mask=attn)

    scores = out.pred_logits.sigmoid()  # (1, 200)
    presence = out.presence_logits.sigmoid()  # (1, 1)
    combined = scores * presence
    best_idx = combined.argmax(dim=1)
    best_score = combined[0, best_idx[0]].item()
    top5 = combined[0].topk(5)

    # Mask stats for best query
    mask = out.pred_masks[0, best_idx[0]]  # (H, W)
    mask_positive = (mask > 0).sum().item()
    mask_total = mask.numel()

    print(f"  {prompt:10s}: best={best_score:.4f} (q{best_idx[0].item():3d}) "
          f"top5=[{', '.join(f'{v:.3f}' for v in top5.values.tolist())}] "
          f"mask={mask_positive}/{mask_total} ({100*mask_positive/mask_total:.0f}%)", flush=True)

print("\nDone!", flush=True)
