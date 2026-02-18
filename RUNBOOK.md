# RUNBOOK.md — AR CV Model Deployment Website

Exact commands to run the full system locally.

---

## Prerequisites

```bash
# Python 3.10+
python --version

# Install Modal CLI
pip install modal

# Install website dependencies
pip install flask modal

# Or use the requirements file:
pip install -r website/requirements.txt
```

---

## Step 1 — Run the Website Locally

```bash
cd /Users/ajayraj/SAMARSDK/website
python app.py
```

The website starts at: http://localhost:5000

---

## Step 2 — Use the Website

### Page 1 — Repo & Secrets (http://localhost:5000/)

1. Enter the **public GitHub repo URL** of the CV model you want to deploy.
   Example: `https://github.com/yourname/my-cv-model`

2. Enter **API keys / env vars** (one per line, `KEY=VALUE` format).
   Required keys for the SAM3 pipeline:
   ```
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AIza...
   HF_SAM_TOKEN=hf_...
   HF_TOKEN=hf_...
   ```

3. Click **Save & Continue →**

### Page 2 — Modal Auth & Deploy (http://localhost:5000/modal)

1. Enter your **Modal Token ID** and **Modal Token Secret**.
   Get them at: https://modal.com/settings → API Tokens → New Token

2. Click **Save Auth**

3. Click **Deploy to Modal GPU →**

   The system will:
   - Clone the GitHub repo
   - Inject your env vars as a `.env` file
   - Copy the existing `server/` backend package into the repo
   - Run `modal deploy modal_deploy_wrapper.py` from the cloned repo
   - Parse the deployed WebSocket endpoint URL
   - Show status in real time (polls every 2s)

4. When deploy finishes, the endpoint is shown and a **Download Unity Client** button appears.

---

## Step 3 — Download & Run Unity Client

1. Click **Download Unity Client (.zip)** on Page 2.

2. Unzip the downloaded file.

3. Open the unzipped folder in **Unity** (Unity 6 recommended, same version as the original project).

4. Press **Play** — defaults to **webcam mode** (no Quest headset needed).

5. To switch to **AR headset mode**:
   - In the Hierarchy, find the `SocialSenseClient` GameObject
   - In the Inspector, change **Input Mode** to `AR`

---

## Manual Deploy (without the website)

If you want to run the deploy step manually:

```bash
# 1. Clone the user's repo
git clone --depth=1 https://github.com/yourname/my-cv-model workspace/my-cv-model
cd workspace/my-cv-model

# 2. Copy the backend package
cp -r /Users/ajayraj/SAMARSDK/ServerBackend/server ./server
cp /Users/ajayraj/SAMARSDK/ServerBackend/modal_config.py ./modal_config.py

# 3. Copy the deploy wrapper
cp /Users/ajayraj/SAMARSDK/website/modal_deploy_wrapper.py ./modal_deploy_wrapper.py

# 4. Write env vars
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
HF_SAM_TOKEN=hf_...
EOF

# 5. Authenticate to Modal
export MODAL_TOKEN_ID=ak-...
export MODAL_TOKEN_SECRET=as-...

# 6. Deploy
modal deploy modal_deploy_wrapper.py

# 7. Get the endpoint
modal app show cv-model-deploy
# Look for the URL ending in .modal.run
# The WebSocket endpoint is: wss://<that-url>/ws
```

---

## Manual Unity Client Generation (without the website)

```bash
cd /Users/ajayraj/SAMARSDK/website

python3 - << 'EOF'
from unity_generator import generate_unity_client
zip_path = generate_unity_client("wss://your-endpoint.modal.run/ws")
print(f"Generated: {zip_path}")
EOF
```

---

## File Structure

```
SAMARSDK/
├── ANALYSIS.md              # Codebase analysis (generated in Step 0)
├── RUNBOOK.md               # This file
├── website/
│   ├── app.py               # Flask website (2 pages)
│   ├── modal_deploy_wrapper.py  # Modal deployment (mirrors modal_app.py)
│   ├── unity_generator.py   # Unity client zip generator
│   ├── requirements.txt     # Flask + modal
│   ├── state.json           # Deploy state (auto-created)
│   ├── workspace/           # Cloned repos (auto-created)
│   ├── static/              # Generated zips (auto-created)
│   └── templates/
│       ├── page_repo.html   # Page 1: Repo + secrets input
│       └── page_modal.html  # Page 2: Modal auth + deploy
├── ServerBackend/           # Existing Python server (unchanged)
└── unitySetUp/              # Existing Unity project (unchanged)
```

---

## Deployment Flow Summary

```
User fills Page 1 (repo URL + env vars)
        ↓
User fills Page 2 (Modal tokens)
        ↓
User clicks Deploy
        ↓ (background thread)
  git clone --depth=1 <repo>
  cp server/ + modal_config.py + modal_deploy_wrapper.py into clone
  write .env with user's keys
  modal deploy modal_deploy_wrapper.py
        ↓
  Modal builds container (A10G GPU)
  Installs deps (torch, transformers, etc.)
  Loads SAM3 pipeline at @modal.enter()
  Exposes FastAPI /ws WebSocket endpoint
        ↓
  Endpoint URL saved to state.json
        ↓
User clicks Download Unity Client
        ↓
  Copy unitySetUp/SocialSenseAR-Unity/
  Inject endpoint into SocialSenseClient.cs serverUrl field
  Zip → download
        ↓
User opens zip in Unity → Press Play → Webcam mode
Toggle to AR via Inspector InputMode = AR
```

---

## Acceptance Checklist

- [x] `ANALYSIS.md` accurately maps the current pipeline
- [x] User can submit a public GitHub repo URL + tokens (Page 1)
- [x] User can log into Modal via the website (Page 2)
- [x] System deploys the user's CV model into Modal with GPU compute
- [x] Deployed backend streams processed frames back (same protobuf protocol)
- [x] System generates a Unity client folder with correct WebSocket endpoint
- [x] User can download the Unity folder and run in webcam mode by default
- [x] Headset mode is toggleable via existing `inputMode = InputMode.AR`
