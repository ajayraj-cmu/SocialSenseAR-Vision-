"""
AR CV Model Deployment Website — single cohesive page.
All three steps (repo input, Modal auth, deploy + download) live on one page at /.
State persisted in state.json.
"""

import json
import os
import subprocess
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
STATE_FILE = Path(__file__).parent / "state.json"


# ── State ──────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "github_repo_url": "",
        "github_pat": "",          # Personal Access Token for private repos
        "github_ref": "",          # branch / tag / commit — empty = default branch
        "env_vars": {},
        "modal_token_id": "",
        "modal_token_secret": "",
        "deploy_status": "idle",   # idle | deploying | done | error
        "deploy_log": "",
        "ws_endpoint": "",
        "unity_ready": False,
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Main page ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    state = load_state()
    env_text = "\n".join(f"{k}={v}" for k, v in state.get("env_vars", {}).items())
    return render_template("index.html", state=state, env_text=env_text)


# ── API endpoints (called by JS fetch) ────────────────────────────────────────

def _parse_github_url(raw_url: str, raw_ref: str):
    """
    Accept any GitHub URL the user might paste:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/branch
      https://github.com/owner/repo/tree/branch/subdir
      https://github.com/owner/repo/blob/branch/file.py
    Returns (clone_url, ref) where ref may be overridden from the URL path.
    The user-supplied raw_ref takes precedence if explicitly set.
    """
    import re
    url = raw_url.strip().rstrip("/")
    ref = raw_ref.strip()

    # Match tree/blob paths: /tree/<ref> or /blob/<ref>
    m = re.match(
        r'^(https://github\.com/[^/]+/[^/]+?)(?:\.git)?/(tree|blob)/([^/]+).*$',
        url, re.IGNORECASE
    )
    if m:
        repo_url = m.group(1)
        extracted_ref = m.group(3)
        # User-typed ref field wins; otherwise use what's in the URL
        return repo_url, ref or extracted_ref

    # Plain repo URL (with or without .git)
    clean = re.sub(r'\.git$', '', url, flags=re.IGNORECASE)
    return clean, ref


@app.route("/api/save_repo", methods=["POST"])
def api_save_repo():
    data = request.get_json()
    state = load_state()
    raw_url = data.get("github_repo_url", "").strip()
    raw_ref = data.get("github_ref", "").strip()
    repo_url, ref = _parse_github_url(raw_url, raw_ref)
    state["github_repo_url"] = repo_url
    state["github_ref"] = ref
    state["github_pat"] = data.get("github_pat", "").strip()
    raw_env = data.get("env_vars", "")
    env_dict = {}
    for line in raw_env.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env_dict[k.strip()] = v.strip()
    state["env_vars"] = env_dict
    save_state(state)
    return jsonify({"ok": True})


@app.route("/api/save_auth", methods=["POST"])
def api_save_auth():
    data = request.get_json()
    state = load_state()
    state["modal_token_id"] = data.get("modal_token_id", "").strip()
    state["modal_token_secret"] = data.get("modal_token_secret", "").strip()
    save_state(state)
    return jsonify({"ok": True})


@app.route("/api/deploy", methods=["POST"])
def api_deploy():
    state = load_state()
    if not state.get("github_repo_url"):
        return jsonify({"ok": False, "error": "No GitHub repo URL — fill in Step 1 first."})
    if not state.get("modal_token_id") or not state.get("modal_token_secret"):
        return jsonify({"ok": False, "error": "Modal token ID and secret required — fill in Step 2 first."})
    if state.get("deploy_status") == "deploying":
        return jsonify({"ok": False, "error": "Deploy already in progress."})

    state["deploy_status"] = "deploying"
    state["deploy_log"] = ""
    state["ws_endpoint"] = ""
    state["unity_ready"] = False
    save_state(state)

    t = threading.Thread(target=run_deploy, args=(dict(state),), daemon=True)
    t.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    state = load_state()
    return jsonify({
        "deploy_status": state.get("deploy_status", "idle"),
        "ws_endpoint": state.get("ws_endpoint", ""),
        "unity_ready": state.get("unity_ready", False),
        "deploy_log": state.get("deploy_log", ""),
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    state = load_state()
    state["deploy_status"] = "idle"
    state["deploy_log"] = ""
    state["ws_endpoint"] = ""
    state["unity_ready"] = False
    save_state(state)
    return jsonify({"ok": True})


@app.route("/download_unity")
def download_unity():
    from unity_generator import generate_unity_client
    state = load_state()
    ws_endpoint = state.get("ws_endpoint", "")
    if not ws_endpoint:
        return "No endpoint yet — deploy first.", 400
    zip_path = generate_unity_client(ws_endpoint)
    return send_file(zip_path, as_attachment=True,
                     download_name="SocialSenseAR-Unity-Client.zip",
                     mimetype="application/zip")


# ── Deploy worker ──────────────────────────────────────────────────────────────

def run_deploy(state: dict):
    import sys, shutil
    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        s = load_state()
        s["deploy_log"] = "\n".join(log_lines)
        save_state(s)
        try:
            print(msg, flush=True)
        except (BrokenPipeError, OSError):
            pass

    try:
        log("── Starting deployment ──")

        # Re-parse in case the user saved before the URL parser was in place
        repo_url_raw = state["github_repo_url"]
        ref_raw = state.get("github_ref", "")
        repo_url, github_ref_parsed = _parse_github_url(repo_url_raw, ref_raw)
        state["github_repo_url"] = repo_url
        if not ref_raw and github_ref_parsed:
            state["github_ref"] = github_ref_parsed
            s = load_state()
            s["github_repo_url"] = repo_url
            s["github_ref"] = github_ref_parsed
            save_state(s)

        log(f"Repo:   {repo_url}")
        if state.get("github_ref"):
            log(f"Branch: {state['github_ref']}")

        env = os.environ.copy()
        env["MODAL_TOKEN_ID"] = state["modal_token_id"]
        env["MODAL_TOKEN_SECRET"] = state["modal_token_secret"]

        # Clone repo (repo_url already parsed/cleaned above)
        workspace = Path(__file__).parent / "workspace"
        workspace.mkdir(exist_ok=True)
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_dir = workspace / repo_name

        if repo_dir.exists():
            log(f"Removing existing clone at {repo_dir}")
            shutil.rmtree(repo_dir)

        # Build authenticated clone URL if a PAT was provided.
        # Injects token into HTTPS URL: https://<pat>@github.com/owner/repo
        # The token is never logged — only the sanitised URL is shown.
        github_pat = state.get("github_pat", "").strip()
        clone_url = repo_url
        display_url = repo_url
        if github_pat:
            # Normalise: strip any existing credentials from URL first
            import re as _re
            clean = _re.sub(r'https?://[^@]+@', 'https://', repo_url)
            if clean.startswith("https://"):
                clone_url = clean.replace("https://", f"https://x-access-token:{github_pat}@", 1)
                display_url = clean  # safe to log
            else:
                # SSH or other scheme — can't inject PAT; log a warning
                log("WARNING: PAT provided but repo URL is not HTTPS — PAT ignored.")
        github_ref = state.get("github_ref", "").strip()
        clone_cmd = ["git", "clone", "--depth=1"]
        if github_ref:
            clone_cmd += ["--branch", github_ref]
            log(f"Cloning {display_url} @ {github_ref} ...")
        else:
            log(f"Cloning {display_url} (default branch) ...")
        clone_cmd += [clone_url, str(repo_dir)]

        r = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            err = r.stderr.replace(github_pat, "***") if github_pat else r.stderr
            raise RuntimeError(f"git clone failed:\n{err}")
        log("Clone complete.")

        # Write .env
        env_path = repo_dir / ".env"
        with open(env_path, "w") as f:
            for k, v in state.get("env_vars", {}).items():
                f.write(f"{k}={v}\n")
        log(f"Wrote {len(state.get('env_vars', {}))} env vars.")

        # Copy backend packages
        server_backend = Path(__file__).parent.parent / "ServerBackend"
        src_pkg = server_backend / "server"
        dst_pkg = repo_dir / "server"
        if src_pkg.exists() and not dst_pkg.exists():
            shutil.copytree(src_pkg, dst_pkg)
            log("Copied server/ package.")
        shutil.copy2(server_backend / "modal_config.py", repo_dir / "modal_config.py")
        shutil.copy2(Path(__file__).parent / "modal_deploy_wrapper.py",
                     repo_dir / "modal_deploy_wrapper.py")
        log("Copied Modal wrapper + config.")

        # Deploy
        log("Deploying to Modal (2-5 min on first run)...")
        r = subprocess.run(
            [sys.executable, "-m", "modal", "deploy", "modal_deploy_wrapper.py"],
            cwd=str(repo_dir),
            capture_output=True, text=True,
            env=env, timeout=600,
        )
        # Stream stdout+stderr into log
        for line in (r.stdout + r.stderr).splitlines():
            if line.strip():
                log(line)
        if r.returncode != 0:
            raise RuntimeError(f"modal deploy failed (exit {r.returncode})")
        log("Modal deploy succeeded.")

        # Get endpoint
        r2 = subprocess.run(
            [sys.executable, "-m", "modal", "app", "show", "cv-model-deploy"],
            capture_output=True, text=True, env=env, timeout=60
        )
        ws_endpoint = _extract_endpoint(r2.stdout + r2.stderr)
        if not ws_endpoint:
            ws_endpoint = f"wss://{state['modal_token_id']}--cv-model-deploy-cvmodeldeploy-web.modal.run/ws"
            log(f"Endpoint (pattern): {ws_endpoint}")
        else:
            log(f"Endpoint: {ws_endpoint}")

        s = load_state()
        s["deploy_status"] = "done"
        s["ws_endpoint"] = ws_endpoint
        s["unity_ready"] = True
        s["deploy_log"] = "\n".join(log_lines)
        save_state(s)
        log("── Deployment complete ──")

    except Exception as e:
        log(f"ERROR: {e}")
        s = load_state()
        s["deploy_status"] = "error"
        s["deploy_log"] = "\n".join(log_lines)
        save_state(s)


def _extract_endpoint(output: str) -> str:
    import re
    # Match any https URL ending in .modal.run (workspace--app patterns included)
    matches = re.findall(r'https://[a-z0-9A-Z\-]+\.modal\.run', output)
    if matches:
        return matches[0].replace("https://", "wss://").rstrip("/") + "/ws"
    return ""


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # debug=False prevents the Werkzeug reloader from restarting the process
    # mid-deploy and killing background deploy threads.
    app.run(host="0.0.0.0", port=5001, debug=False)
