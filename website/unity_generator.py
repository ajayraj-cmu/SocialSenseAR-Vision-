"""
Unity Client Generator
======================

Takes the existing unitySetUp/ project and produces a downloadable .zip
with the WebSocket endpoint injected into SocialSenseClient.cs.

Only one line changes in the Unity project:
  SocialSenseClient.cs line ~43:
    public string serverUrl = "<injected-modal-ws-url>";

Default mode remains InputMode.Webcam (no change needed — already default).

Returns path to the generated .zip file.
"""

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


UNITY_SRC = (
    Path(__file__).parent.parent
    / "unity-client"
)

# The single line in SocialSenseClient.cs that holds the server URL
SERVER_URL_PATTERN = re.compile(
    r'(public\s+string\s+serverUrl\s*=\s*")[^"]*(";)'
)

# Folders/files to exclude from the zip (generated/cache dirs — large and unnecessary)
EXCLUDE_DIRS = {
    "Library", "Temp", "Logs", "obj", ".vs",
    "__pycache__", ".git",
}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo"}


def generate_unity_client(ws_endpoint: str) -> Path:
    """
    Copy unitySetUp/SocialSenseAR-Unity to a temp dir,
    inject ws_endpoint into SocialSenseClient.cs,
    zip it up, return path to zip.
    """
    # Output zip goes in website/static/
    out_dir = Path(__file__).parent / "static"
    out_dir.mkdir(exist_ok=True)
    out_zip = out_dir / "SocialSenseAR-Unity-Client.zip"

    # Work in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dest = tmp_path / "SocialSenseAR-Unity"

        print(f"[unity_generator] Copying Unity project to {dest}...")
        shutil.copytree(
            str(UNITY_SRC),
            str(dest),
            ignore=_ignore_fn,
        )

        # Inject the WebSocket endpoint
        client_script = dest / "Assets" / "Scripts" / "SocialSenseClient.cs"
        if client_script.exists():
            _inject_endpoint(client_script, ws_endpoint)
            print(f"[unity_generator] Injected endpoint: {ws_endpoint}")
        else:
            print(f"[unity_generator] WARNING: SocialSenseClient.cs not found at {client_script}")

        # Write a README into the zip root
        readme = dest / "ENDPOINT_README.txt"
        readme.write_text(
            f"SocialSenseAR Unity Client\n"
            f"==========================\n\n"
            f"WebSocket Endpoint: {ws_endpoint}\n\n"
            f"Quick Start:\n"
            f"1. Extract this zip to your desired location (e.g. Desktop, Documents)\n"
            f"2. Double-click OpenInUnity.command (Mac) or OpenInUnity.bat (Windows)\n"
            f"   to open the project in Unity and load the final scene.\n"
            f"3. Press Play — defaults to Webcam mode\n"
            f"4. To switch to AR headset mode:\n"
            f"   Select the SocialSenseClient GameObject in the scene,\n"
            f"   change Input Mode to 'AR' in the Inspector.\n\n"
            f"The serverUrl field in SocialSenseClient.cs has been pre-filled:\n"
            f"  {ws_endpoint}\n"
        )

        # OpenInUnity script (macOS) — opens project in Unity, loads final scene
        open_mac = dest / "OpenInUnity.command"
        open_mac.write_text(
            "#!/bin/bash\n"
            'cd "$(dirname "$0")"\n'
            '# Try Unity app first, then Unity Hub\n'
            'if open -a "Unity" . 2>/dev/null; then\n'
            '  echo "Opening in Unity..."\n'
            'elif open -a "Unity Hub" . 2>/dev/null; then\n'
            '  echo "Opening in Unity Hub..."\n'
            'else\n'
            '  echo "Unity not found. Please install Unity and open this folder manually."\n'
            '  echo "Project path: $(pwd)"\n'
            '  open .\n'
            'fi\n',
            encoding="utf-8",
        )
        open_mac.chmod(0o755)

        # Editor script to open final scene when project loads
        editor_dir = dest / "Assets" / "Editor"
        editor_dir.mkdir(exist_ok=True)
        open_scene_script = editor_dir / "OpenFinalSceneOnLoad.cs"
        open_scene_script.write_text(
            "// Auto-generated: opens final scene when project loads (first open).\n"
            "#if UNITY_EDITOR\n"
            "using UnityEditor;\n"
            "using UnityEditor.SceneManagement;\n"
            "\n"
            "static class OpenFinalSceneOnLoad {\n"
            "    [InitializeOnLoadMethod]\n"
            "    static void OnLoad() {\n"
            "        EditorApplication.delayCall += () => {\n"
            "            var active = EditorSceneManager.GetActiveScene();\n"
            "            if (string.IsNullOrEmpty(active.path))\n"
            "                EditorSceneManager.OpenScene(\"Assets/final.unity\");\n"
            "        };\n"
            "    }\n"
            "}\n"
            "#endif\n",
            encoding="utf-8",
        )

        # OpenInUnity script (Windows)
        open_win = dest / "OpenInUnity.bat"
        open_win.write_text(
            '@echo off\n'
            'cd /d "%~dp0"\n'
            'set "UNITY="\n'
            'for /d %%D in ("C:\\Program Files\\Unity\\Hub\\Editor\\*") do set "UNITY=%%D\\Editor\\Unity.exe"\n'
            'if not exist "%UNITY%" set "UNITY=C:\\Program Files\\Unity\\Editor\\Unity.exe"\n'
            'if exist "%UNITY%" (\n'
            '  start "" "%UNITY%" -projectPath "%CD%"\n'
            '  echo Opening in Unity...\n'
            ') else (\n'
            '  echo Unity not found. Please install Unity and open this folder manually.\n'
            '  echo Project path: %CD%\n'
            '  explorer .\n'
            ')\n'
            'pause\n',
            encoding="utf-8",
        )

        # Zip the dest folder
        print(f"[unity_generator] Zipping to {out_zip}...")
        with zipfile.ZipFile(str(out_zip), "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in dest.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmp_path)
                    zf.write(str(file_path), str(arcname))

        print(f"[unity_generator] Done: {out_zip}")

    return out_zip


def _ignore_fn(src, names):
    """shutil.copytree ignore function — skips Library/, Temp/, etc."""
    ignored = set()
    for name in names:
        if name in EXCLUDE_DIRS:
            ignored.add(name)
        elif any(name.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
            ignored.add(name)
    return ignored


def _inject_endpoint(script_path: Path, ws_endpoint: str):
    """Replace the serverUrl value in SocialSenseClient.cs."""
    text = script_path.read_text(encoding="utf-8")

    # Replace existing serverUrl value
    new_text = SERVER_URL_PATTERN.sub(
        r'\g<1>' + ws_endpoint + r'\g<2>',
        text,
    )

    if new_text == text:
        # Pattern didn't match — try a more lenient replacement
        # Look for any line with "serverUrl" and a string assignment
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "serverUrl" in line and "=" in line and '"' in line:
                # Replace the quoted string value on this line
                lines[i] = re.sub(r'"[^"]*"', f'"{ws_endpoint}"', line, count=1)
                break
        new_text = "\n".join(lines)

    script_path.write_text(new_text, encoding="utf-8")
