# Deploy SocialSense to Modal

One-time setup, then deploy. If segmentation returns 0 segments, the usual cause is a missing or invalid HuggingFace token.

## 1. Modal auth and secret

```bash
pip install modal
modal setup
```

Create the secret (required for segmentation):

```bash
modal secret create socialsense-secrets \
  HF_SAM_TOKEN=hf_your_token_here
```

- Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) (gated model). Without access, model load will fail with 401.

Optional (voice / scene understanding):

```bash
modal secret create socialsense-secrets \
  HF_SAM_TOKEN=hf_... \
  OPENAI_API_KEY=sk-... \
  GEMINI_API_KEY=...
```

If the secret already exists, update it in [Modal Dashboard](https://modal.com) → Secrets → `socialsense-secrets`.

## 2. Deploy

From the **SocialSenseAR** directory (the one that contains `modal_app.py`, `server/`, and `config/`):

```bash
cd /path/to/SocialSenseAR
modal deploy modal_app.py
```

Use the printed WebSocket URL in Unity (e.g. `wss://.../ws`).

## 3. If it doesn’t segment

- **Backend up but 0 segments**  
  - Check Modal app logs: open the deployment in [Modal Dashboard](https://modal.com/apps) and look at logs.
  - If you see `HF_SAM_TOKEN: MISSING` or a 401 from HuggingFace, add or fix `HF_SAM_TOKEN` in the `socialsense-secrets` secret and redeploy.
  - If the container fails to start, the logs will show the pipeline init error (often missing token or no access to `facebook/sam3`).

- **Cold start**  
  First request after deploy or after scale-to-zero can take ~60–90s while the SAM3 model loads. After that, frames should segment normally.
