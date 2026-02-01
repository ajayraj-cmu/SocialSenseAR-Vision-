"""Export SAM3 model components to ONNX for ORT+TensorRT acceleration.

Exports the vision encoder and decoder as separate ONNX models.
The segmenter loads these with ONNX Runtime + TensorRT EP at startup.

Usage:
    python -m server.vision.sam3_export

Outputs (in project root):
    sam3_vision.onnx   — vision encoder (pixel_values → 9 flat tensors)
    sam3_decoder.onnx  — decoder pipeline (vision tensors + text → masks + logits)
    sam3_meta.json     — shapes, preprocessing params, output names
"""

import os
import sys
import json
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ── Vision Encoder ──────────────────────────────────────────────────────────

def export_vision_encoder(model, processor, device, output_dir):
    """Export vision encoder to ONNX with flat tensor outputs.

    Sam3VisionEncoderOutput contains 9 tensors:
      last_hidden_state          (1, 5184, 1024)
      fpn_hidden_states[0..3]    (1, 256, H, W) at 4 scales
      fpn_position_encoding[0..3](1, 256, H, W) at 4 scales

    We flatten these into a 9-element tuple for clean ONNX export.
    """
    import torch
    from PIL import Image

    logger.info("=" * 60)
    logger.info("Exporting VISION ENCODER to ONNX")
    logger.info("=" * 60)

    # Sample input
    dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    inputs = processor(images=dummy_img, text="person", return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device == "cuda":
        pixel_values = pixel_values.half()

    pv_shape = list(pixel_values.shape)
    logger.info(f"  pixel_values: {pv_shape} {pixel_values.dtype}")

    # Wrapper: flatten structured output → tuple of tensors
    class VisionWrapper(torch.nn.Module):
        def __init__(self, vision_encoder):
            super().__init__()
            self.vision_encoder = vision_encoder

        def forward(self, pixel_values):
            vis = self.vision_encoder(pixel_values)
            # Flatten: lhs, fpn_hs[0..3], fpn_pe[0..3]
            return (
                vis.last_hidden_state,
                vis.fpn_hidden_states[0], vis.fpn_hidden_states[1],
                vis.fpn_hidden_states[2], vis.fpn_hidden_states[3],
                vis.fpn_position_encoding[0], vis.fpn_position_encoding[1],
                vis.fpn_position_encoding[2], vis.fpn_position_encoding[3],
            )

    wrapper = VisionWrapper(model.vision_encoder)
    wrapper.eval()

    # Get sample output shapes
    with torch.no_grad():
        sample_out = wrapper(pixel_values)

    output_names = [
        "last_hidden_state",
        "fpn_hs_0", "fpn_hs_1", "fpn_hs_2", "fpn_hs_3",
        "fpn_pe_0", "fpn_pe_1", "fpn_pe_2", "fpn_pe_3",
    ]

    output_shapes = {}
    for name, tensor in zip(output_names, sample_out):
        output_shapes[name] = list(tensor.shape)
        logger.info(f"  {name}: {list(tensor.shape)} {tensor.dtype}")

    # Export in FP32 on CPU (GPU VRAM too small for FP32 tracing of 32-layer ViT)
    onnx_path = os.path.join(output_dir, "sam3_vision.onnx")
    logger.info(f"  Moving vision encoder to CPU for FP32 ONNX export...")
    t0 = time.perf_counter()

    # Move to CPU + FP32 for tracing (FP16 tracing not well-supported by ONNX)
    orig_device = next(model.parameters()).device
    orig_dtype = next(model.parameters()).dtype
    model.cpu().float()
    torch.cuda.empty_cache()

    wrapper_f32 = VisionWrapper(model.vision_encoder)
    wrapper_f32.eval()
    pv_f32 = pixel_values.cpu().float()

    dynamic_axes = {"pixel_values": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    logger.info(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper_f32,
        (pv_f32,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    export_ms = (time.perf_counter() - t0) * 1000
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"  Exported: {file_size:.1f} MB, {export_ms:.0f}ms")

    # Restore model to original device/dtype
    model.to(device=orig_device, dtype=orig_dtype)
    torch.cuda.empty_cache()

    # Quick validation with CPU ORT
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"pixel_values": pv_f32.numpy()})
        logger.info(f"  ONNX validation OK: {len(ort_out)} outputs, "
                     f"shapes={[o.shape for o in ort_out]}")
        del sess
    except Exception as e:
        logger.warning(f"  ONNX validation failed: {e}")

    return onnx_path, pv_shape, output_shapes, output_names


# ── Decoder Pipeline ────────────────────────────────────────────────────────

def export_decoder(model, processor, device, output_dir, max_token_len):
    """Export decoder pipeline to ONNX.

    Inputs: 4 vision tensors (fpn_hs_0..2, fpn_pe_2) + text_embeds + attention_mask
    Outputs: pred_masks, pred_logits, presence_logits

    Replicates model.forward() decoder path but with flat tensor inputs,
    avoiding the Sam3VisionEncoderOutput structured type.
    """
    import torch
    from PIL import Image

    logger.info("")
    logger.info("=" * 60)
    logger.info("Exporting DECODER to ONNX")
    logger.info("=" * 60)

    # Get sample vision embeddings
    dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    inputs = processor(images=dummy_img, text="person", return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device == "cuda":
        pixel_values = pixel_values.half()

    with torch.no_grad():
        vis = model.get_vision_features(pixel_values=pixel_values)

    # Get text_embeds (pre-computed, bypasses CLIP)
    with torch.no_grad():
        text_out = model.get_text_features(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            return_dict=True,
        )
        text_embeds = text_out.pooler_output.detach()

    # Pad attention_mask to max_token_len
    attention_mask = inputs["attention_mask"].to(device)
    pad_len = max_token_len - attention_mask.shape[1]
    if pad_len > 0:
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
        text_embeds_padded = torch.nn.functional.pad(text_embeds, (0, 0, 0, pad_len), value=0)
    else:
        text_embeds_padded = text_embeds

    # The decoder uses: fpn_hidden_states[:-1] = [0,1,2], fpn_position_encoding[2]
    fpn_hs_0 = vis.fpn_hidden_states[0]
    fpn_hs_1 = vis.fpn_hidden_states[1]
    fpn_hs_2 = vis.fpn_hidden_states[2]
    fpn_pe_2 = vis.fpn_position_encoding[2]

    logger.info(f"  fpn_hs_0: {list(fpn_hs_0.shape)}")
    logger.info(f"  fpn_hs_1: {list(fpn_hs_1.shape)}")
    logger.info(f"  fpn_hs_2: {list(fpn_hs_2.shape)}")
    logger.info(f"  fpn_pe_2: {list(fpn_pe_2.shape)}")
    logger.info(f"  text_embeds: {list(text_embeds_padded.shape)}")
    logger.info(f"  attention_mask: {list(attention_mask.shape)}")

    # Wrapper: replicates model.forward() decoder path with flat inputs
    # Reference: modeling_sam3.py Sam3Model.forward lines 66-183
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.detr_encoder = model.detr_encoder
            self.detr_decoder = model.detr_decoder
            self.dot_product_scoring = model.dot_product_scoring
            self.mask_decoder = model.mask_decoder

        def forward(self, fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2,
                    text_embeds, attention_mask):
            fpn_hidden_states = [fpn_hs_0, fpn_hs_1, fpn_hs_2]
            text_mask = attention_mask.bool()

            # DETR encoder: vision + text → fused features
            encoder_outputs = self.detr_encoder(
                vision_features=[fpn_hs_2],
                text_features=text_embeds,
                vision_pos_embeds=[fpn_pe_2],
                text_mask=text_mask,
            )

            # DETR decoder: cross-attend queries to encoder output
            decoder_outputs = self.detr_decoder(
                vision_features=encoder_outputs.last_hidden_state,
                text_features=encoder_outputs.text_features,
                vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
                text_mask=text_mask,
                spatial_shapes=encoder_outputs.spatial_shapes,
            )

            # Scoring (dot product between decoder queries and text)
            all_pred_logits = self.dot_product_scoring(
                decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
                text_features=encoder_outputs.text_features,
                text_mask=text_mask,
            ).squeeze(-1)

            pred_logits = all_pred_logits[-1]           # (B, 200)
            decoder_hs = decoder_outputs.intermediate_hidden_states[-1]
            presence_logits = decoder_outputs.presence_logits[-1]  # (B, 1)

            # Mask decoder: generate pixel-level masks
            mask_outputs = self.mask_decoder(
                decoder_queries=decoder_hs,
                backbone_features=fpn_hidden_states,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                prompt_features=text_embeds,
                prompt_mask=text_mask,
            )

            return mask_outputs.pred_masks, pred_logits, presence_logits

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    # Verify wrapper produces correct output
    with torch.no_grad():
        test_out = wrapper(fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2,
                           text_embeds_padded, attention_mask)

    output_names = ["pred_masks", "pred_logits", "presence_logits"]
    output_info = {}
    for name, tensor in zip(output_names, test_out):
        output_info[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        logger.info(f"  Output '{name}': {list(tensor.shape)} {tensor.dtype}")

    # Export in FP32 on CPU (same reason as vision: VRAM too small for FP32 tracing)
    onnx_path = os.path.join(output_dir, "sam3_decoder.onnx")
    logger.info(f"  Moving decoder to CPU for FP32 ONNX export...")
    t0 = time.perf_counter()

    orig_device = next(model.parameters()).device
    orig_dtype = next(model.parameters()).dtype
    model.cpu().float()
    torch.cuda.empty_cache()

    wrapper_f32 = DecoderWrapper(model)
    wrapper_f32.eval()

    input_names = ["fpn_hs_0", "fpn_hs_1", "fpn_hs_2", "fpn_pe_2",
                   "text_embeds", "attention_mask"]

    # Dynamic axes: batch on everything
    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    sample_inputs = (
        fpn_hs_0.cpu().float(), fpn_hs_1.cpu().float(),
        fpn_hs_2.cpu().float(), fpn_pe_2.cpu().float(),
        text_embeds_padded.cpu().float(), attention_mask.cpu().long(),
    )

    logger.info(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper_f32,
        sample_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    export_ms = (time.perf_counter() - t0) * 1000
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"  Exported: {file_size:.1f} MB, {export_ms:.0f}ms")

    # Restore model
    model.to(device=orig_device, dtype=orig_dtype)
    torch.cuda.empty_cache()

    # Quick validation
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feed = {name: t.float().numpy() if name != "attention_mask"
                else t.numpy().astype(np.int64)
                for name, t in zip(input_names, sample_inputs)}
        ort_out = sess.run(None, feed)
        logger.info(f"  ONNX validation OK: {[o.shape for o in ort_out]}")
        del sess
    except Exception as e:
        logger.warning(f"  ONNX validation failed: {e}")

    decoder_input_shapes = {
        "fpn_hs_0": list(fpn_hs_0.shape),
        "fpn_hs_1": list(fpn_hs_1.shape),
        "fpn_hs_2": list(fpn_hs_2.shape),
        "fpn_pe_2": list(fpn_pe_2.shape),
        "text_embeds": list(text_embeds_padded.shape),
        "attention_mask": list(attention_mask.shape),
    }

    return onnx_path, output_info, output_names, decoder_input_shapes


# ── Top-1 Decoder (scoring + best-query mask only) ─────────────────────────

def export_topk_decoder(model, processor, device, output_dir, max_token_len):
    """Export decoder that selects best query and generates only 1 mask per prompt.

    Same inputs as full decoder, but internally:
    1. Runs DETR encoder + decoder + scoring (same)
    2. Picks the best query via argmax (new)
    3. Runs mask decoder with only 1 query (200x less mask computation)

    Outputs: best_mask (B, 1, H, W), best_score (B,)
    """
    import torch
    from PIL import Image

    logger.info("")
    logger.info("=" * 60)
    logger.info("Exporting TOP-1 DECODER to ONNX")
    logger.info("=" * 60)

    # Get sample tensors (same setup as export_decoder)
    dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    inputs = processor(images=dummy_img, text="person", return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device == "cuda":
        pixel_values = pixel_values.half()

    with torch.no_grad():
        vis = model.get_vision_features(pixel_values=pixel_values)
        text_out = model.get_text_features(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            return_dict=True,
        )
        text_embeds = text_out.pooler_output.detach()

    attention_mask = inputs["attention_mask"].to(device)
    pad_len = max_token_len - attention_mask.shape[1]
    if pad_len > 0:
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
        text_embeds_padded = torch.nn.functional.pad(text_embeds, (0, 0, 0, pad_len), value=0)
    else:
        text_embeds_padded = text_embeds

    fpn_hs_0 = vis.fpn_hidden_states[0]
    fpn_hs_1 = vis.fpn_hidden_states[1]
    fpn_hs_2 = vis.fpn_hidden_states[2]
    fpn_pe_2 = vis.fpn_position_encoding[2]

    class TopKDecoderWrapper(torch.nn.Module):
        """Decoder that selects best query and generates 1 mask instead of 200."""
        def __init__(self, model):
            super().__init__()
            self.detr_encoder = model.detr_encoder
            self.detr_decoder = model.detr_decoder
            self.dot_product_scoring = model.dot_product_scoring
            self.mask_decoder = model.mask_decoder

        def forward(self, fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2,
                    text_embeds, attention_mask):
            fpn_hidden_states = [fpn_hs_0, fpn_hs_1, fpn_hs_2]
            text_mask = attention_mask.bool()

            # DETR encoder
            encoder_outputs = self.detr_encoder(
                vision_features=[fpn_hs_2],
                text_features=text_embeds,
                vision_pos_embeds=[fpn_pe_2],
                text_mask=text_mask,
            )

            # DETR decoder
            decoder_outputs = self.detr_decoder(
                vision_features=encoder_outputs.last_hidden_state,
                text_features=encoder_outputs.text_features,
                vision_pos_encoding=encoder_outputs.pos_embeds_flattened,
                text_mask=text_mask,
                spatial_shapes=encoder_outputs.spatial_shapes,
            )

            # Scoring
            all_pred_logits = self.dot_product_scoring(
                decoder_hidden_states=decoder_outputs.intermediate_hidden_states,
                text_features=encoder_outputs.text_features,
                text_mask=text_mask,
            ).squeeze(-1)

            pred_logits = all_pred_logits[-1]           # (B, 200)
            decoder_hs = decoder_outputs.intermediate_hidden_states[-1]  # (B, 200, 256)
            presence_logits = decoder_outputs.presence_logits[-1]  # (B, 1)

            # Select best query
            scores = pred_logits.sigmoid() * presence_logits.sigmoid()  # (B, 200)
            best_idx = scores.argmax(dim=1)  # (B,)
            best_score = scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # (B,)

            # Gather best query: (B, 200, 256) → (B, 1, 256)
            idx = best_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, decoder_hs.shape[2])
            decoder_hs_best = torch.gather(decoder_hs, 1, idx)

            # Mask decoder with just 1 query (200x less mask computation)
            mask_outputs = self.mask_decoder(
                decoder_queries=decoder_hs_best,
                backbone_features=fpn_hidden_states,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                prompt_features=text_embeds,
                prompt_mask=text_mask,
            )

            return mask_outputs.pred_masks, best_score  # (B,1,H,W), (B,)

    wrapper = TopKDecoderWrapper(model)
    wrapper.eval()

    # Verify
    with torch.no_grad():
        test_out = wrapper(fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2,
                           text_embeds_padded, attention_mask)

    output_names = ["best_mask", "best_score"]
    for name, tensor in zip(output_names, test_out):
        logger.info(f"  Output '{name}': {list(tensor.shape)} {tensor.dtype}")

    # Export in FP32 on CPU
    onnx_path = os.path.join(output_dir, "sam3_topk_decoder.onnx")
    logger.info(f"  Moving to CPU for FP32 ONNX export...")
    t0 = time.perf_counter()

    orig_device = next(model.parameters()).device
    orig_dtype = next(model.parameters()).dtype
    model.cpu().float()
    import torch as _torch
    _torch.cuda.empty_cache()

    wrapper_f32 = TopKDecoderWrapper(model)
    wrapper_f32.eval()

    input_names = ["fpn_hs_0", "fpn_hs_1", "fpn_hs_2", "fpn_pe_2",
                   "text_embeds", "attention_mask"]

    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    sample_inputs = (
        fpn_hs_0.cpu().float(), fpn_hs_1.cpu().float(),
        fpn_hs_2.cpu().float(), fpn_pe_2.cpu().float(),
        text_embeds_padded.cpu().float(), attention_mask.cpu().long(),
    )

    logger.info(f"  Exporting to {onnx_path}...")
    _torch.onnx.export(
        wrapper_f32,
        sample_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    export_ms = (time.perf_counter() - t0) * 1000
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"  Exported: {file_size:.1f} MB, {export_ms:.0f}ms")

    # Restore model
    model.to(device=orig_device, dtype=orig_dtype)
    _torch.cuda.empty_cache()

    # Quick validation
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feed = {name: t.float().numpy() if name != "attention_mask"
                else t.numpy().astype(np.int64)
                for name, t in zip(input_names, sample_inputs)}
        ort_out = sess.run(None, feed)
        logger.info(f"  ONNX validation OK: {[o.shape for o in ort_out]}")
        del sess
    except Exception as e:
        logger.warning(f"  ONNX validation failed: {e}")

    return onnx_path


# ── Metadata ────────────────────────────────────────────────────────────────

def save_metadata(output_dir, processor, pv_shape, vis_output_shapes,
                  vis_output_names, dec_output_info, dec_output_names,
                  dec_input_shapes, max_token_len):
    """Save metadata JSON for the runtime to reconstruct tensors."""
    ip = processor.image_processor
    size = getattr(ip, 'size', {})
    if isinstance(size, dict):
        if 'height' in size and 'width' in size:
            image_size = [size['height'], size['width']]
        elif 'longest_edge' in size:
            image_size = [size['longest_edge'], size['longest_edge']]
        else:
            image_size = list(pv_shape[2:])
    elif isinstance(size, (list, tuple)):
        image_size = list(size)
    else:
        image_size = list(pv_shape[2:])

    meta = {
        "pixel_values_shape": pv_shape,
        "vision_output_shapes": vis_output_shapes,
        "vision_output_names": vis_output_names,
        "decoder_output_info": dec_output_info,
        "decoder_output_names": dec_output_names,
        "decoder_input_shapes": dec_input_shapes,
        "max_token_len": max_token_len,
        "image_size": image_size,
        "image_mean": getattr(ip, 'image_mean', [0.485, 0.456, 0.406]),
        "image_std": getattr(ip, 'image_std', [0.229, 0.224, 0.225]),
        "do_rescale": getattr(ip, 'do_rescale', True),
    }

    meta_path = os.path.join(output_dir, "sam3_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"\nMetadata saved: {meta_path}")
    return meta_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import torch
    from server.config import ServerConfig

    config = ServerConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = PROJECT_ROOT

    logger.info("SAM3 ONNX Export for TensorRT Acceleration")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {config.sam3_model}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("")

    from transformers import Sam3Model, Sam3Processor

    logger.info("Loading SAM3 model and processor...")
    t0 = time.perf_counter()
    processor = Sam3Processor.from_pretrained(config.sam3_model)
    model = Sam3Model.from_pretrained(config.sam3_model)
    model.to(device).eval()
    if device == "cuda":
        model.half()
    logger.info(f"Model loaded: {(time.perf_counter() - t0)*1000:.0f}ms")

    # Discover max token length
    from PIL import Image
    from server.vision.sam3_segmenter import ALL_PROMPTS

    dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    max_token_len = 0
    for prompt in ALL_PROMPTS:
        inputs = processor(images=dummy_img, text=prompt, return_tensors="pt")
        max_token_len = max(max_token_len, inputs["input_ids"].shape[1])
    logger.info(f"Max token length: {max_token_len}")

    # Export vision encoder
    vis_onnx, pv_shape, vis_shapes, vis_names = export_vision_encoder(
        model, processor, device, output_dir)

    # Export full decoder (for fallback)
    dec_onnx, dec_info, dec_names, dec_input_shapes = export_decoder(
        model, processor, device, output_dir, max_token_len)

    # Export top-1 decoder (scoring + best-query mask only — 200x less mask work)
    topk_onnx = export_topk_decoder(
        model, processor, device, output_dir, max_token_len)

    # Save metadata
    save_metadata(output_dir, processor, pv_shape, vis_shapes,
                  vis_names, dec_info, dec_names, dec_input_shapes, max_token_len)

    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Vision:       {vis_onnx}")
    logger.info(f"  Decoder:      {dec_onnx}")
    logger.info(f"  Top-1 Decoder:{topk_onnx}")
    logger.info("")
    logger.info("Next: run build_trt_engine.py to build TRT engines from ONNX files.")


if __name__ == "__main__":
    main()
