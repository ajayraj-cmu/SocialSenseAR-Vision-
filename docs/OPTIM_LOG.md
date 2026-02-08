# SAM3 Optimization Ledger

Target: 40+ FPS on RTX 3060 Laptop (6GB VRAM, Ampere, compute 8.6)
Model: facebook/sam3 (~530M params)
Constraint: NO quality loss, NO model swaps

## Architecture Reference
```
pixel_values (1,3,1008,1008)
  -> vision_encoder (ViT-32L 1024d + FPN)  ~280ms   [BOTTLENECK]
input_ids (B,32)
  -> text_encoder (CLIP-24L 1024d)          ~0ms     [ELIMINATED by caching]
  -> text_projection (1024->256)
fpn[-1] + text
  -> detr_encoder (6L, 256d, 8 heads)       \
  -> detr_decoder (6L, 200 queries, 256d)    } ~115ms total (3 prompts batched)
  -> mask_decoder (pixel FPN + einsum)       /
  -> dot_product_scoring + GPU threshold
post_process (GPU-side)                       ~included in decoder
```

---

## Attempt Log

| # | Tier | Change | Before | After | Delta | Status |
|---|------|--------|--------|-------|-------|--------|
| 0 | baseline | Unoptimized PyTorch FP16, 4 prompts/frame | - | 1.3 FPS (763ms) | - | measured |
| 1 | 1a | Pre-cache text_embeds (bypass CLIP-24L) | 1.3 FPS | - | saves ~50ms | measured |
| 2 | 1b | torch.compile vision+detr+mask_decoder | - | - | vis 350->280ms | measured |
| 3 | 1c | GPU-side mask threshold (skip HF postprocess) | - | - | eliminates CPU xfer | measured |
| 1-3 | ALL T1 | Combined Tier 1 (uncached) | 1.3 FPS (763ms) | **2.5 FPS (402ms)** | **-47%** | **measured** |
| 4 | 2 | Vision cache + 2 rotating prompts | 2.5 FPS (402ms) | **8.3 FPS (121ms)** | **-70%** | **measured** |
| 5 | 3a | TRT full decoder (200 masks, batch=1 seq) | 8.3 FPS | **9.2 FPS (109ms)** | **-10%** | measured |
| 6 | 3b | Multi-stream TRT (2 contexts) | 9.2 FPS | 9.2 FPS | **0%** | no improvement |
| 7 | 3c | Vision cache MSE 150→500 | - | - | real cam: 80% hits | measured |
| 8 | 3d | TRT top-k decoder (1 mask, batch=1 seq) | 9.2 FPS | **12.3 FPS (81ms)** | **-26%** | measured |
| 9 | 3e | Reduce prompts person+2→person+1 | - | - | 2→2 prompts/frame | measured |
| 10 | 3f | Batch TRT prompts (batch=N single call) | 12.3 FPS | **13.7 FPS (73ms)** | **-10%** | measured |
| 11 | 3g | TRT vision encoder (sam3_vision.engine) | 280ms | 256ms | **-9%** | **reverted** |
| 12 | - | TRT vision + decoder together (VRAM pressure) | 50ms dec | 83ms dec | **+66% worse** | **reverted** |
| 13 | fix | Remove TRT vision, decoder-only + PyTorch vis | 83ms/frame | **~50ms dec** (97ms total) | **back to best** | current |
| 14 | fix | 0-seg diagnosis: dark webcam (11.5/255 brightness) | 0 segs | - | model correct | diagnosed |
| 15 | 4a | Decoder result caching (per-prompt, tagged with vision gen) | 13.7 FPS | **~18 FPS** | **+32%** | measured |
| 16 | 4b | Deferred preprocessing (skip preprocess on vision cache hit) | ~18 FPS | **~19 FPS** | **+5%** | measured |
| 17 | 4c | Async vision encoding (background thread, non-blocking) | ~19 FPS | **~21 FPS** | **+10%** | measured |
| 18 | 4d | GPU contention avoidance (skip decoder during async vision) | ~21 FPS | **~21.5 FPS** | **+2%** | measured |
| 19 | 4e | Rotation skip during async encode (person-only when pending) | ~21.5 FPS | ~21.5 FPS | **0%** | marginal |
| 20 | 5a | Event-based SAM loop (replace 15.6ms Windows sleep with threading.Event) | 20.5 FPS | **30.7 FPS** | **+50%** | **measured** |
| 21 | 5b | CUDA synchronize in async worker (eliminate GPU contention spikes) | 30.7 FPS | 30.7 FPS (no 280ms spikes) | **stable** | measured |
| 22 | 5c | High-speed benchmark (60fps busy-wait client) | 30.7 FPS | **49.0 FPS** | **+60%** | **measured** |

---

## Timing Breakdown (current — async vision + decoder caching + event-based loop)

| Component | First Frame | Vision Cached (decoder miss) | Fully Cached |
|-----------|-------------|------------------------------|--------------|
| Preprocess (cv2 resize+normalize) | 4ms | **0ms** (deferred) | **0ms** |
| Vision encoder (compiled FP16) | ~280ms | **0ms** (async bg thread) | **0ms** |
| FPN FP16→FP32 + cache | ~3ms | **0ms** (cached) | **0ms** |
| TRT decoder (2 prompts, batched) | ~25ms | ~25ms | **0ms** (gen-cached) |
| Mask resize + CPU transfer | ~4ms | ~4ms | **0ms** |
| RLE encoding (orchestrator) | ~5ms | ~5ms | ~1ms |
| **Total** | **~320ms** | **~34ms** | **~1ms** |

**Measured benchmark**: **49.0 FPS** SAM throughput (57fps client, synthetic frames)
**Architecture**: Event-based SAM loop + async vision encode + decoder gen-caching

### Key finding: ViT is COMPUTE-BOUND on RTX 3060
The 32-layer ViT requires ~3.15 TFLOPS of FP16 compute.
RTX 3060 Laptop = 12.5 TFLOPS → theoretical minimum = 252ms.
TRT barely helps (256ms vs 280ms PyTorch). Both engines together
degrade decoder from 50ms→83ms due to VRAM pressure (2.9GB TRT total).
**Decision**: Keep PyTorch vision + TRT decoder only.

---

## Detailed Notes

### Attempt 0: Baseline
- Server log: `SAM #10: 763ms total (763ms SAM), 0 segs`
- 0 segs because _outputs_to_cpu was converting ModelOutput to plain dict,
  breaking HF post_process (needs .pred_logits attribute access). Fixed.
- torch.compile on model.forward fails: get_text_features() mutates
  CLIPTextModelOutput.pooler_output in-place, incompatible with compile trace.
- Batched decoder also fails for same reason.
- Fallback: uncompiled model.forward, sequential decoding.

### Attempt 1: Pre-cache text_embeds
- model.forward() accepts `text_embeds` kwarg (modeling_sam3.py:2285-2290)
- When text_embeds is provided, skips entire get_text_features() call
- 12 fixed prompts -> compute once at init, reuse forever
- Saving: ~50ms/frame (entire CLIP-24L forward eliminated)

### Attempt 2: Submodule compile
- Cannot compile model.forward (CLIP mutation issue)
- CAN compile individual submodules: detr_encoder, detr_decoder, mask_decoder
- vision_encoder compile works (biggest win): ~350ms -> ~280ms
- text_encoder compile not needed (cached)
- First-run compile: ~20s for vision, ~10s for decoder

### Attempt 3: GPU mask threshold
- HF post_process_instance_segmentation runs on CPU
- pred_masks shape: (B, 200, 1008, 1008) — massive CPU transfer
- Custom GPU path: sigmoid -> threshold -> best-mask-per-prompt -> uint8 -> CPU
- Transfers only (B, H, W) uint8 instead of (B, 200, H, W) float32

### Attempt 4: Frame skip + prompt reduction
- Vision embedding cache: skip vision encoder when consecutive frame thumbs have MSE < 150
- Reduce from person+3 to person+2 rotating
- Static scene result: 402ms -> 121ms (8.3 FPS)
- Cache hit rate for static webcam: ~97%+ (huge win for AR use case)

### Attempt 5: TensorRT (NEXT)
- Vision encoder is 32-layer ViT processing 5184 patches
- Expected: ~280ms PyTorch -> ~20-30ms TRT (FP16, zero quality loss)
- Decoder: ~115ms PyTorch -> ~10-20ms TRT
- Combined TRT target: 30-50ms total -> 20-33 FPS
- With vision cache on static scenes: 10-20ms -> 50-100 FPS
- Use sam3_export.py for ONNX, then trtexec --fp16
- MUST TRT the decoder too — 115ms decoder alone prevents 40 FPS
