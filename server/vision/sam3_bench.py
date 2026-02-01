"""SAM3 component-level benchmark + optimization plan.

Profiles every submodule independently, then applies optimizations
incrementally and measures the delta. Produces a clear report.

Usage:
    python -m server.vision.sam3_bench

Key discoveries from model analysis:
1. model.forward() accepts `text_embeds` kwarg — bypasses 24-layer CLIP
   text encoder entirely.  Since our prompts are FIXED ("person", "chair",
   etc.), we can pre-compute text embeddings ONCE and never run CLIP again.
   This alone may save 30-80ms per forward call.

2. torch.compile fails on full model.forward because get_text_features()
   mutates CLIPTextModelOutput.pooler_output in-place.  But individual
   submodules (detr_encoder, detr_decoder, mask_decoder) should compile fine.

3. The vision encoder (32-layer ViT @ 1008x1008 → 5184 patches) is the
   bottleneck at ~200-350ms.  torch.compile already helps here.

4. The DETR encoder flattens 3 FPN levels into 110k tokens of cross-attention
   — that's the second bottleneck.

Architecture (facebook/sam3):
    pixel_values (1,3,1008,1008)
        → vision_encoder (ViT-32L + FPN)     → fpn_hidden_states (3 levels)
    input_ids (B,32)
        → text_encoder (CLIP-24L)            → text_features (B,32,256)
        → text_projection (1024→256)
    fpn[-1] + text_features
        → detr_encoder (6L, 256d)            → encoder_out
    encoder_out
        → detr_decoder (6L, 200 queries)     → decoder_out
    decoder_out + fpn
        → mask_decoder (pixel FPN + einsum)  → pred_masks (B,200,H,W)
        → dot_product_scoring                → pred_logits (B,200)
"""

import os
import sys
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def gpu_sync_time(torch):
    """Synchronize CUDA and return perf_counter."""
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, torch, warmup=3, repeats=10, label=""):
    """Benchmark a callable with CUDA sync. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        t0 = gpu_sync_time(torch)
        fn()
        t1 = gpu_sync_time(torch)
        times.append((t1 - t0) * 1000)

    med = sorted(times)[len(times) // 2]
    mn = min(times)
    mx = max(times)
    log.info(f"  {label:40s}  median={med:7.1f}ms  min={mn:7.1f}ms  max={mx:7.1f}ms")
    return med


def main():
    import torch
    from transformers import Sam3Model, Sam3Processor
    from PIL import Image
    from server.config import ServerConfig

    config = ServerConfig()
    device = "cuda"

    log.info("=" * 70)
    log.info("SAM3 COMPONENT BENCHMARK + OPTIMIZATION PLAN")
    log.info("=" * 70)
    log.info(f"Device: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    log.info(f"Model:  {config.sam3_model}")
    log.info("")

    # ----------------------------------------------------------------
    # 1. Load model
    # ----------------------------------------------------------------
    log.info("Loading model...")
    t0 = time.perf_counter()
    processor = Sam3Processor.from_pretrained(config.sam3_model)
    model = Sam3Model.from_pretrained(config.sam3_model)
    model.to(device).eval().half()
    log.info(f"Loaded in {(time.perf_counter() - t0)*1000:.0f}ms")
    log.info("")

    # ----------------------------------------------------------------
    # 2. Prepare inputs
    # ----------------------------------------------------------------
    dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    inputs = processor(images=dummy_img, text="person", return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device).half()
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Batched text (4 prompts)
    prompts = ["person", "chair", "table", "desk"]
    batch_ids_list = []
    batch_attn_list = []
    max_len = 0
    for p in prompts:
        inp = processor(images=dummy_img, text=p, return_tensors="pt")
        batch_ids_list.append(inp["input_ids"].to(device))
        batch_attn_list.append(inp["attention_mask"].to(device))
        max_len = max(max_len, inp["input_ids"].shape[1])

    # Pad to uniform length
    for i in range(len(prompts)):
        pad = max_len - batch_ids_list[i].shape[1]
        if pad > 0:
            batch_ids_list[i] = torch.nn.functional.pad(batch_ids_list[i], (0, pad))
            batch_attn_list[i] = torch.nn.functional.pad(batch_attn_list[i], (0, pad))

    batch_ids = torch.cat(batch_ids_list, dim=0)      # (4, max_len)
    batch_attn = torch.cat(batch_attn_list, dim=0)     # (4, max_len)

    log.info(f"pixel_values: {list(pixel_values.shape)} {pixel_values.dtype}")
    log.info(f"input_ids:    {list(input_ids.shape)}")
    log.info(f"batch_ids:    {list(batch_ids.shape)} ({len(prompts)} prompts)")
    log.info("")

    # ================================================================
    # PHASE 1: BASELINE — profile each component
    # ================================================================
    log.info("=" * 70)
    log.info("PHASE 1: COMPONENT-LEVEL PROFILING (uncompiled, FP16)")
    log.info("=" * 70)

    # 1a. Vision encoder
    with torch.no_grad():
        vision_out = model.get_vision_features(pixel_values=pixel_values)

    vis_ms = bench(
        lambda: model.get_vision_features(pixel_values=pixel_values),
        torch, label="vision_encoder (ViT-32L + FPN)"
    )

    # 1b. Text encoder (single prompt)
    with torch.no_grad():
        text_out = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_embeds_single = text_out.pooler_output  # (1, seq, 256)

    txt_ms = bench(
        lambda: model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, return_dict=True),
        torch, label="text_encoder (CLIP-24L, 1 prompt)"
    )

    # 1c. Text encoder (batched 4 prompts)
    txt_batch_ms = bench(
        lambda: model.get_text_features(input_ids=batch_ids, attention_mask=batch_attn, return_dict=True),
        torch, label="text_encoder (CLIP-24L, 4 prompts)"
    )

    # 1d. Full forward (single prompt, vision_embeds pre-computed)
    def full_fwd_single():
        return model(
            vision_embeds=vision_out,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    with torch.no_grad():
        single_out = full_fwd_single()

    fwd_single_ms = bench(full_fwd_single, torch, label="model.forward (1 prompt, vision cached)")

    # 1e. Full forward (4 prompts batched, vision_embeds expanded)
    def expand_vis(vis, B):
        fields = {}
        for k in vis.keys():
            v = vis[k]
            if isinstance(v, torch.Tensor):
                fields[k] = v.expand(B, *v.shape[1:])
            elif isinstance(v, (list, tuple)):
                fields[k] = type(v)(
                    t.expand(B, *t.shape[1:]) if isinstance(t, torch.Tensor) else t for t in v
                )
            else:
                fields[k] = v
        return type(vis)(**fields)

    vis_4 = expand_vis(vision_out, 4)

    def full_fwd_batch():
        return model(
            vision_embeds=vis_4,
            input_ids=batch_ids,
            attention_mask=batch_attn,
        )

    fwd_batch_ms = bench(full_fwd_batch, torch, label="model.forward (4 prompts, vision cached)")

    # 1f. Full forward with PRE-COMPUTED text_embeds (bypass CLIP)
    with torch.no_grad():
        batch_text_out = model.get_text_features(input_ids=batch_ids, attention_mask=batch_attn, return_dict=True)
        batch_text_embeds = batch_text_out.pooler_output  # (4, seq, 256)

    def full_fwd_text_cached():
        return model(
            vision_embeds=vis_4,
            text_embeds=batch_text_embeds,
            attention_mask=batch_attn,
        )

    fwd_textcache_ms = bench(full_fwd_text_cached, torch, label="model.forward (4p, vis+text cached)")

    log.info("")
    log.info("-" * 70)
    text_saving = fwd_batch_ms - fwd_textcache_ms
    log.info(f"TEXT CACHE SAVING: {text_saving:.1f}ms per frame ({fwd_batch_ms:.1f} → {fwd_textcache_ms:.1f})")
    total_baseline = vis_ms + fwd_batch_ms
    total_textcache = vis_ms + fwd_textcache_ms
    log.info(f"BASELINE TOTAL:   {total_baseline:.1f}ms = {1000/total_baseline:.1f} FPS")
    log.info(f"TEXT-CACHED TOTAL: {total_textcache:.1f}ms = {1000/total_textcache:.1f} FPS")
    log.info("-" * 70)
    log.info("")

    # ================================================================
    # PHASE 2: SUBMODULE torch.compile
    # ================================================================
    log.info("=" * 70)
    log.info("PHASE 2: torch.compile ON SUBMODULES")
    log.info("=" * 70)

    # 2a. vision_encoder (already known to work)
    log.info("\nCompiling vision_encoder (default mode)...")
    model.vision_encoder = torch.compile(model.vision_encoder)
    # Warmup compile
    with torch.no_grad():
        model.get_vision_features(pixel_values=pixel_values)
    vis_compiled_ms = bench(
        lambda: model.get_vision_features(pixel_values=pixel_values),
        torch, label="vision_encoder COMPILED"
    )

    # Update vision_out for compiled encoder
    with torch.no_grad():
        vision_out = model.get_vision_features(pixel_values=pixel_values)
    vis_4 = expand_vis(vision_out, 4)

    # 2b. detr_encoder
    log.info("\nCompiling detr_encoder...")
    try:
        model.detr_encoder = torch.compile(model.detr_encoder)
        # Need to run a full forward to trigger compile on detr_encoder
        with torch.no_grad():
            model(vision_embeds=vis_4, text_embeds=batch_text_embeds, attention_mask=batch_attn)
        fwd_detr_enc_ms = bench(
            full_fwd_text_cached, torch, label="forward (detr_encoder compiled)"
        )
        log.info(f"  Delta: {fwd_textcache_ms - fwd_detr_enc_ms:+.1f}ms")
    except Exception as e:
        log.info(f"  FAILED: {e}")
        fwd_detr_enc_ms = fwd_textcache_ms

    # 2c. detr_decoder
    log.info("\nCompiling detr_decoder...")
    try:
        model.detr_decoder = torch.compile(model.detr_decoder)
        with torch.no_grad():
            model(vision_embeds=vis_4, text_embeds=batch_text_embeds, attention_mask=batch_attn)
        fwd_detr_dec_ms = bench(
            full_fwd_text_cached, torch, label="forward (detr_enc+dec compiled)"
        )
        log.info(f"  Delta: {fwd_detr_enc_ms - fwd_detr_dec_ms:+.1f}ms")
    except Exception as e:
        log.info(f"  FAILED: {e}")
        fwd_detr_dec_ms = fwd_detr_enc_ms

    # 2d. mask_decoder
    log.info("\nCompiling mask_decoder...")
    try:
        model.mask_decoder = torch.compile(model.mask_decoder)
        with torch.no_grad():
            model(vision_embeds=vis_4, text_embeds=batch_text_embeds, attention_mask=batch_attn)
        fwd_all_compiled_ms = bench(
            full_fwd_text_cached, torch, label="forward (ALL submodules compiled)"
        )
        log.info(f"  Delta: {fwd_detr_dec_ms - fwd_all_compiled_ms:+.1f}ms")
    except Exception as e:
        log.info(f"  FAILED: {e}")
        fwd_all_compiled_ms = fwd_detr_dec_ms

    log.info("")
    log.info("-" * 70)
    final_total = vis_compiled_ms + fwd_all_compiled_ms
    log.info(f"FULLY OPTIMIZED:  {final_total:.1f}ms = {1000/final_total:.1f} FPS")
    log.info(f"  vision:  {vis_compiled_ms:.1f}ms")
    log.info(f"  forward: {fwd_all_compiled_ms:.1f}ms (text cached + all compiled)")
    log.info(f"  speedup: {total_baseline/final_total:.2f}x over baseline")
    log.info("-" * 70)
    log.info("")

    # ================================================================
    # PHASE 3: REDUCED PROMPT COUNT
    # ================================================================
    log.info("=" * 70)
    log.info("PHASE 3: PROMPT COUNT SWEEP")
    log.info("=" * 70)

    for n_prompts in [1, 2, 3, 4]:
        ids_n = batch_ids[:n_prompts]
        attn_n = batch_attn[:n_prompts]

        with torch.no_grad():
            text_n = model.get_text_features(input_ids=ids_n, attention_mask=attn_n, return_dict=True).pooler_output
        vis_n = expand_vis(vision_out, n_prompts)

        def fwd_n(v=vis_n, t=text_n, a=attn_n):
            return model(vision_embeds=v, text_embeds=t, attention_mask=a)

        ms = bench(fwd_n, torch, warmup=3, repeats=8,
                   label=f"{n_prompts} prompt(s) (text cached, compiled)")
        total = vis_compiled_ms + ms
        log.info(f"    → Total: {total:.1f}ms = {1000/total:.1f} FPS")
        log.info("")

    # ================================================================
    # PHASE 4: POST-PROCESSING
    # ================================================================
    log.info("=" * 70)
    log.info("PHASE 4: POST-PROCESSING BENCHMARK")
    log.info("=" * 70)

    with torch.no_grad():
        sample_out = model(vision_embeds=vis_4, text_embeds=batch_text_embeds, attention_mask=batch_attn)

    # Move to CPU (simulates _outputs_to_cpu)
    def to_cpu():
        cpu_fields = {}
        for k in sample_out.keys():
            v = sample_out[k]
            if hasattr(v, 'cpu'):
                cpu_fields[k] = v.cpu().float()
            else:
                cpu_fields[k] = v
        return type(sample_out)(**cpu_fields)

    cpu_ms = bench(to_cpu, torch, warmup=2, repeats=5, label="outputs → CPU (4 prompts)")

    cpu_out = to_cpu()
    def postprocess():
        return processor.post_process_instance_segmentation(
            cpu_out, target_sizes=[(480, 640)] * 4, threshold=0.3
        )

    try:
        pp_ms = bench(postprocess, torch, warmup=2, repeats=5, label="post_process_instance_segmentation")
    except Exception as e:
        log.info(f"  post_process FAILED: {e}")
        pp_ms = 0

    log.info("")

    # ================================================================
    # SUMMARY
    # ================================================================
    log.info("=" * 70)
    log.info("OPTIMIZATION SUMMARY")
    log.info("=" * 70)
    log.info("")
    log.info("Baseline (uncompiled, no text cache):")
    log.info(f"  vision={vis_ms:.0f}ms + forward={fwd_batch_ms:.0f}ms + post={cpu_ms + pp_ms:.0f}ms")
    log.info(f"  TOTAL: {vis_ms + fwd_batch_ms + cpu_ms + pp_ms:.0f}ms = {1000/(vis_ms + fwd_batch_ms + cpu_ms + pp_ms):.1f} FPS")
    log.info("")
    log.info("Optimized (compiled vision + text cache + compiled decoder):")
    log.info(f"  vision={vis_compiled_ms:.0f}ms + forward={fwd_all_compiled_ms:.0f}ms + post={cpu_ms + pp_ms:.0f}ms")
    opt_total = vis_compiled_ms + fwd_all_compiled_ms + cpu_ms + pp_ms
    log.info(f"  TOTAL: {opt_total:.0f}ms = {1000/opt_total:.1f} FPS")
    log.info("")
    log.info("KEY FINDINGS:")
    log.info(f"  1. Pre-cache text embeddings:  saves ~{text_saving:.0f}ms/frame (bypass CLIP-24L)")
    log.info(f"  2. torch.compile vision:       {vis_ms:.0f}ms → {vis_compiled_ms:.0f}ms")
    log.info(f"  3. torch.compile detr+mask:    {fwd_textcache_ms:.0f}ms → {fwd_all_compiled_ms:.0f}ms")
    log.info(f"  4. CPU transfer + postprocess: {cpu_ms + pp_ms:.0f}ms (optimize if >10ms)")
    log.info("")
    log.info("NEXT STEPS FOR 40+ FPS:")
    log.info("  - Apply text_embeds caching in sam3_segmenter.py")
    log.info("  - Compile detr_encoder, detr_decoder, mask_decoder individually")
    log.info("  - If still <40 FPS: export to ONNX/TensorRT (sam3_export.py)")
    log.info("  - If postprocess is slow: do mask thresholding on GPU before CPU transfer")
    log.info("  - Reduce prompts_per_frame if needed (2 rotating = person + 2)")


if __name__ == "__main__":
    import torch
    with torch.no_grad():
        main()
