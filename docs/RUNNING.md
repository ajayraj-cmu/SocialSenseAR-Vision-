# Running This Project (Including With Your Modal Setup)

This doc explains how someone else can run the app **on their computer** after you push the repo to GitHub, including using **your exact Modal secret** if you want.

---

## What Is Safe to Push to GitHub

- **Do push:** All code, `modal_app.py`, `modal_config.py`, `requirements.txt`, `.env.example`, this file.
- **Do not push:** `.env` (it’s in `.gitignore`). Never commit real API keys or tokens.

---

## What the App Needs to Run on Modal

1. **Modal authentication**  
   So the CLI can run `modal deploy` / `modal run` / `modal serve`. That uses a **Modal token** (tied to a Modal account).

2. **Modal Secret named `socialsense-secrets`**  
   The app expects a secret with the following keys:
   - `HF_SAM_TOKEN` — Hugging Face token for gated SAM3 model (`facebook/sam3`). **Required.**
   - `HF_PERSONAPLEX_TOKEN` — Hugging Face token for PersonaPlex model (`nvidia/personaplex-7b-v1`). Required only if PersonaPlex is enabled.
   - `HF_TOKEN` — Legacy fallback. Used if `HF_SAM_TOKEN` or `HF_PERSONAPLEX_TOKEN` are not set.
   - `OPENAI_API_KEY` (optional, for Whisper/voice transcription)
   - `GEMINI_API_KEY` (optional, for Gemini Vision scene understanding and command planning)
   - `PERSONAPLEX_ENABLED` — Set to `true` to enable PersonaPlex voice backend (default: `false`)
   - `PERSONAPLEX_URL` — PersonaPlex WebSocket URL (default: `ws://localhost:8998/api/chat`)

   At startup, the Modal container logs `SET` or `MISSING` for each token so you can verify.

Those can come from **your** Modal account or from **theirs**.

---

## Option A: They Use Your Exact Modal Secret (Your Account)

They run the app under **your** Modal account, so they automatically use your existing secret and billing.

**What you give them (securely, e.g. 1Password / secure chat):**

- **Modal token**  
  From your machine:
  ```bash
  modal token show
  ```
  Or from [Modal Dashboard](https://modal.com/settings) → Token. They need both **Token ID** and **Token Secret**.

**What they do once:**

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Use your token (replace with the values you gave them)
modal token set --token-id <TOKEN_ID> --token-secret <TOKEN_SECRET>

# Deploy and run
modal deploy modal_app.py
python tools/webcam_modal_client.py --url wss://<your-app-url>/ws --show
```

**Caveat:** Anyone with this token has full access to your Modal account (billing, secrets, apps). Only share with people you trust; consider Option B for others.

---

## Option B: They Use Their Own Modal Account (Same API Keys)

They use **their** Modal account and create a secret that has the **same keys** as yours (so the app behaves the same).

**What you give them (securely):**

- The **values** of: `HF_TOKEN`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (from your `.env` or from Modal Dashboard → Secrets → `socialsense-secrets`).  
- You do **not** give them your Modal token.

**What they do once:**

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# 1) Create their own Modal account and log in
modal setup
# (follow the browser flow; they get their own token)

# 2) Create a secret with the same name and keys you provided
modal secret create socialsense-secrets \
  HF_SAM_TOKEN=<value-you-sent> \
  HF_PERSONAPLEX_TOKEN=<value-you-sent> \
  OPENAI_API_KEY=<value-you-sent> \
  GEMINI_API_KEY=<value-you-sent>

# 3) Deploy and run (their deployment URL will be under their account)
modal deploy modal_app.py
# Use the WebSocket URL printed or from Modal Dashboard for their app
python tools/webcam_modal_client.py --url wss://<their-app-url>/ws --show
```

No access to your Modal account; they only use the same API keys for the app.

---

## Summary: What to Give Them

| Goal | What you give | What they run |
|------|----------------|----------------|
| **Use your exact Modal secret** (your account) | Your **Modal token** (ID + secret) | `modal token set ...` then `modal deploy` |
| **Same behavior, their account** | **API key values** (HF, OpenAI, Gemini) | `modal setup`, `modal secret create socialsense-secrets ...`, then `modal deploy` |

In both cases they run the app **directly on their computer** (client + browser to Modal); the only difference is whose Modal account and token they use, and whether you share a token or only API key values.
