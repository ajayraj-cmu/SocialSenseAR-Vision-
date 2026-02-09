"""Compare TRT decoder vs PyTorch decoder on the same inputs."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

# Capture webcam frame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
cap.release()
print(f"Frame: {frame.shape}", flush=True)

# Load model
from server.config import ServerConfig
config = ServerConfig()
device = "cuda"

from transformers import Sam3Model, Sam3Processor
processor = Sam3Processor.from_pretrained(config.sam3_model)
model = Sam3Model.from_pretrained(config.sam3_model).to(device).eval().half()

# Preprocess
from PIL import Image
pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
inputs = processor(images=pil_img, text="person", return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device).half()

# Run vision encoder
with torch.no_grad():
    vis = model.get_vision_features(pixel_values=pixel_values)

# Get text embeddings
with torch.no_grad():
    text_out = model.get_text_features(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        return_dict=True,
    )
    text_embeds = text_out.pooler_output.detach()

# Pad to 32 tokens
attn = inputs["attention_mask"].to(device)
pad_len = 32 - attn.shape[1]
if pad_len > 0:
    attn = torch.nn.functional.pad(attn, (0, pad_len), value=0)
    text_embeds = torch.nn.functional.pad(text_embeds, (0, 0, 0, pad_len), value=0)

# ---- PyTorch forward ----
print("\n--- PyTorch decoder ---", flush=True)
with torch.no_grad():
    pt_out = model(
        vision_embeds=vis,
        text_embeds=text_embeds,
        attention_mask=attn,
    )
pt_scores = pt_out.pred_logits.sigmoid()  # (1, 200)
pt_presence = pt_out.presence_logits.sigmoid()  # (1, 1)
pt_combined = pt_scores * pt_presence
best_idx = pt_combined.argmax(dim=1)
best_score_pt = pt_combined[0, best_idx[0]].item()
print(f"  best_score = {best_score_pt:.4f} (query {best_idx[0].item()})", flush=True)
print(f"  pred_logits range: [{pt_out.pred_logits.min():.2f}, {pt_out.pred_logits.max():.2f}]", flush=True)
print(f"  presence_logits: {pt_out.presence_logits[0,0].item():.4f}", flush=True)

# Get the mask
best_mask_pt = pt_out.pred_masks[0, best_idx[0]]  # (H, W)
print(f"  mask shape: {best_mask_pt.shape}, sum: {best_mask_pt.sum().item():.0f}", flush=True)
mask_area = (best_mask_pt > 0).sum().item()
print(f"  mask area (>0): {mask_area}", flush=True)

# ---- TRT decoder ----
print("\n--- TRT decoder ---", flush=True)
fpn_hs_0 = vis.fpn_hidden_states[0].float().contiguous()
fpn_hs_1 = vis.fpn_hidden_states[1].float().contiguous()
fpn_hs_2 = vis.fpn_hidden_states[2].float().contiguous()
fpn_pe_2 = vis.fpn_position_encoding[2].float().contiguous()
te_f32 = text_embeds.float().contiguous()
am_i64 = attn.long().contiguous()

print(f"  fpn_hs_0: {fpn_hs_0.shape} range=[{fpn_hs_0.min():.3f}, {fpn_hs_0.max():.3f}]", flush=True)
print(f"  fpn_hs_2: {fpn_hs_2.shape} range=[{fpn_hs_2.min():.3f}, {fpn_hs_2.max():.3f}]", flush=True)
print(f"  fpn_pe_2: {fpn_pe_2.shape} range=[{fpn_pe_2.min():.3f}, {fpn_pe_2.max():.3f}]", flush=True)
print(f"  te_f32: {te_f32.shape} range=[{te_f32.min():.3f}, {te_f32.max():.3f}]", flush=True)
print(f"  am_i64: {am_i64.shape} values={am_i64[0].tolist()}", flush=True)

# Load TRT engine
import tensorrt as trt
sp_dir = os.path.dirname(os.path.dirname(torch.__file__))
trt_libs = os.path.join(sp_dir, 'tensorrt_libs')
if os.path.isdir(trt_libs):
    os.environ['PATH'] = trt_libs + ';' + os.environ.get('PATH', '')

engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_topk_decoder.engine")
print(f"  Loading engine: {engine_path}", flush=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
stream = torch.cuda.Stream()

# Set shapes
ctx.set_input_shape("fpn_hs_0", [1, 256, 288, 288])
ctx.set_input_shape("fpn_hs_1", [1, 256, 144, 144])
ctx.set_input_shape("fpn_hs_2", [1, 256, 72, 72])
ctx.set_input_shape("fpn_pe_2", [1, 256, 72, 72])
ctx.set_input_shape("text_embeds", [1, 32, 256])
ctx.set_input_shape("attention_mask", [1, 32])

# Set addresses
ctx.set_tensor_address("fpn_hs_0", fpn_hs_0.data_ptr())
ctx.set_tensor_address("fpn_hs_1", fpn_hs_1.data_ptr())
ctx.set_tensor_address("fpn_hs_2", fpn_hs_2.data_ptr())
ctx.set_tensor_address("fpn_pe_2", fpn_pe_2.data_ptr())
ctx.set_tensor_address("text_embeds", te_f32.data_ptr())
ctx.set_tensor_address("attention_mask", am_i64.data_ptr())

out_mask = torch.empty(1, 1, 288, 288, dtype=torch.float32, device=device)
out_score = torch.empty(1, dtype=torch.float32, device=device)
ctx.set_tensor_address("best_mask", out_mask.data_ptr())
ctx.set_tensor_address("best_score", out_score.data_ptr())

ok = ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
print(f"  TRT ok: {ok}", flush=True)
print(f"  best_score = {out_score.item():.4f}", flush=True)
print(f"  mask shape: {out_mask.shape}, sum: {out_mask.sum().item():.0f}", flush=True)
mask_area_trt = (out_mask > 0).sum().item()
print(f"  mask area (>0): {mask_area_trt}", flush=True)

print(f"\n--- Comparison ---", flush=True)
print(f"  PyTorch best_score: {best_score_pt:.4f}", flush=True)
print(f"  TRT best_score:     {out_score.item():.4f}", flush=True)
print(f"  PyTorch mask area:  {mask_area}", flush=True)
print(f"  TRT mask area:      {mask_area_trt}", flush=True)
print("Done!", flush=True)
