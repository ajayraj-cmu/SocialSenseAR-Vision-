#!/usr/bin/env python3
"""
SAM + Gemini Voice-Controlled Environment Modifier
ACCESSIBILITY AID FOR SENSORY REGULATION

Main entry point that integrates all refactored modules.

VISUAL MODES:
1. Visual Noise Cancellation - Blur distracting elements (screens, lights, people)
2. Color Remapping - Change colors of objects to reduce stimulation
3. Motion Dampening - Reduce motion salience for predictability

USAGE:
- Say "hey vibe" to start recording, then speak your command
- End with "thanks" to process the command
- Example: "hey vibe" → "blur my face" → "thanks"
"""
import os
import time
import cv2

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, '.env'),
        os.path.join(script_dir, 'scripts', '.env'),
    ]

    for env_path in possible_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            break

# Load .env file before importing modules
load_env_file()

# Import refactored modules
from src.gemini import GeminiAgent
from src.audio import AudioProcessor, VoiceListener
from src.control import EnvironmentController
from src.camera import AsyncCamera


def main():
    """Main application loop."""
    # Initialize components
    print("\nInitializing components...")
    gemini_agent = GeminiAgent()
    voice_listener = VoiceListener()
    audio_processor = AudioProcessor()

    # Start audio stream if available
    if audio_processor.available:
        audio_processor.start_audio_stream()

    # Initialize environment controller with dependencies
    controller = EnvironmentController(gemini_agent, voice_listener, audio_processor)

    # Use async camera for smooth streaming
    camera = AsyncCamera(0, 640, 480, 30)

    if not camera.is_opened():
        print("❌ No camera!")
        return

    # Wait for first frame
    time.sleep(0.5)

    fps_times = []
    display_fps_times = []
    last_display_time = time.time()

    try:
        while True:
            # Get latest frame from async camera
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Track display FPS
            now = time.time()
            display_fps_times.append(now - last_display_time)
            last_display_time = now
            if len(display_fps_times) > 30:
                display_fps_times.pop(0)
            display_fps = 1.0 / (sum(display_fps_times) / len(display_fps_times)) if display_fps_times else 30

            # Process frame
            t0 = time.time()
            display = controller.process_frame(frame)
            elapsed = time.time() - t0

            # Processing FPS
            fps_times.append(elapsed)
            if len(fps_times) > 20:
                fps_times.pop(0)
            process_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 30

            # UI overlay
            h, w = display.shape[:2]

            # Only show full UI if NOT in clean view mode
            if not controller.clean_view_mode:
                # Status bar
                cv2.rectangle(display, (0, 0), (w, 65), (30, 30, 30), -1)
                cv2.putText(display, f"Display: {display_fps:.0f} FPS | Process: {process_fps:.0f} | Objects: {len(controller.masks)}",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(display, voice_listener.status,
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if voice_listener.last_command:
                    cv2.putText(display, f"Last: {voice_listener.last_command[:40]}",
                               (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # Active effects panel
                y = 75
                cv2.putText(display, f"Labels: {len(controller.masks)} | Effects: {len(controller.active_effects)}",
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                if controller.active_effects:
                    y += 18
                    cv2.putText(display, "ACTIVE EFFECTS:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    for label, color in list(controller.active_effects.items())[:6]:
                        y += 14
                        cv2.putText(display, f"  {label} -> {color}", (10, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)

                # Help text at bottom
                help_y = h - 25
                cv2.rectangle(display, (0, help_y - 5), (w, h), (30, 30, 30), -1)
                cv2.putText(display, "Say 'hey vibe' to start | V=clean view | C=clear | L=labels | S=save | Q=quit",
                           (10, help_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            else:
                # Clean view: minimal UI
                cv2.putText(display, f"[CLEAN VIEW] V=toggle | {len(controller.active_effects)} effects",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.imshow("Voice Environment Controller", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                controller.clean_view_mode = not controller.clean_view_mode
                mode_str = "CLEAN (effects only)" if controller.clean_view_mode else "FULL (labels + borders)"
                print(f"  👁️ View mode: {mode_str}")
            elif key == ord('c'):
                controller.clear_effects()
            elif key == ord('p'):
                controller.autopilot_enabled = not controller.autopilot_enabled
                print(f"  🤖 Autopilot = {'ON' if controller.autopilot_enabled else 'OFF'}")
            elif key == ord('s'):
                fn = f"voice_env_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 Saved {fn}")
            elif key == ord('l'):
                labels = [m[1] for m in controller.masks]
                print(f"\n{'='*50}")
                print("📋 ALL AVAILABLE LABELS:")
                for i, label in enumerate(labels):
                    effect = controller.active_effects.get(label, None)
                    status = f" -> {effect}" if effect else ""
                    print(f"  {i+1}. {label}{status}")
                print(f"{'='*50}\n")

            # Auto-stop when the online training loop reaches target confidence
            if gemini_agent.should_stop():
                print(f"🏁 Stopping (auto-train complete): {gemini_agent.stop_reason()}")
                break

    finally:
        controller.close()
        camera.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
