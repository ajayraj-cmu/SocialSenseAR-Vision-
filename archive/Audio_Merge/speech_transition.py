import pyaudio
import webrtcvad
import time
import json
import threading
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 3
SILENCE_THRESHOLD = 20
MIN_SPEECH_FRAMES = 8
SILENCE_TIMEOUT_SECONDS = 2.0  # Time before showing yellow when other person stops

# State file path
STATE_FILE = Path(__file__).parent / "transition_state.json"

# ============================================================================
# STATE FILE MANAGEMENT
# ============================================================================
def update_transition_state(status):
    """Write transition state to JSON file"""
    try:
        state = {
            "status": status,
            "timestamp": time.time()
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error writing state file: {e}")

# ============================================================================
# COLOR STATUS FUNCTIONS
# ============================================================================
def print_status(status, current_time=None):
    """Print color-coded status"""
    if status == 'green':
        print('\033[92m🟢 GREEN - User speaking\033[0m')
    elif status == 'blue':
        print('\033[94m🔵 BLUE - Other person speaking\033[0m')
    elif status == 'yellow':
        print('\033[93m🟡 YELLOW - Other person stopped (2+ sec silence)\033[0m')

# ============================================================================
# WEBVIEW UI LAUNCH
# ============================================================================
def start_webview_ui():
    """Start the transition indicator webview in a separate thread"""
    try:
        from ui_transition_indicator import start_ui
        start_ui()
    except Exception as e:
        print(f"Warning: Could not start UI: {e}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    # Start webview UI in separate thread
    webview_thread = threading.Thread(target=start_webview_ui, daemon=True)
    webview_thread.start()
    
    # Give webview time to initialize
    time.sleep(1)
    
    # Device selection
    p = pyaudio.PyAudio()
    print("\n" + "="*60)
    print("AVAILABLE MICROPHONES")
    print("="*60)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            default = " (DEFAULT)" if i == p.get_default_input_device_info()['index'] else ""
            print(f"  [{i}] {info['name']}{default}")

    print("\n" + "="*60)
    mic1_input = input("Mic 1 (User) - Enter index or press Enter for default: ").strip()
    mic1_index = int(mic1_input) if mic1_input else p.get_default_input_device_info()['index']
    mic1_name = p.get_device_info_by_index(mic1_index)['name']

    mic2_input = input("Mic 2 (Other Person) - Enter index: ").strip()
    mic2_index = int(mic2_input) if mic2_input else p.get_default_input_device_info()['index']
    mic2_name = p.get_device_info_by_index(mic2_index)['name']

    print("\n" + "="*60)
    print(f"Mic 1: [{mic1_index}] {mic1_name} (User)")
    print(f"Mic 2: [{mic2_index}] {mic2_name} (Other Person)")
    print("="*60 + "\n")

    # Setup VAD and streams
    vad_mic1 = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    vad_mic2 = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    stream_mic1 = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                         input=True, input_device_index=mic1_index,
                         frames_per_buffer=FRAME_SIZE)

    stream_mic2 = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                         input=True, input_device_index=mic2_index,
                         frames_per_buffer=FRAME_SIZE)

    # State tracking
    state_mic1 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}
    state_mic2 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0, 
                  'last_speech_time': None, 'speech_end_time': None, 'was_speaking': False}
    
    current_status = None
    last_status_print_time = 0
    status_print_interval = 0.1  # Print status every 100ms

    print("🎤 Listening... (Press Ctrl+C to stop)\n")
    print("Status colors:")
    print("  🟢 GREEN - User speaking")
    print("  🔵 BLUE - Other person speaking")
    print("  🟡 YELLOW - Other person stopped (2+ sec silence)\n")

    try:
        while True:
            current_time = time.time()
            
            # Process Mic 1 (User)
            frame1 = stream_mic1.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech1 = vad_mic1.is_speech(frame1, SAMPLE_RATE)
            
            if is_speech1:
                if not state_mic1['speaking']:
                    state_mic1['speaking'] = True
                state_mic1['frames'].append(frame1)
                state_mic1['speech_count'] += 1
                state_mic1['silence'] = 0
            elif state_mic1['speaking']:
                state_mic1['silence'] += 1
                state_mic1['frames'].append(frame1)
                
                if state_mic1['silence'] >= SILENCE_THRESHOLD:
                    if state_mic1['speech_count'] >= MIN_SPEECH_FRAMES:
                        state_mic1['speaking'] = False
                    state_mic1 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}
            
            # Process Mic 2 (Other Person)
            frame2 = stream_mic2.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech2 = vad_mic2.is_speech(frame2, SAMPLE_RATE)
            
            if is_speech2:
                if not state_mic2['speaking']:
                    state_mic2['speaking'] = True
                    state_mic2['was_speaking'] = True
                    # Reset speech_end_time when they start speaking again
                    state_mic2['speech_end_time'] = None
                state_mic2['frames'].append(frame2)
                state_mic2['speech_count'] += 1
                state_mic2['silence'] = 0
                state_mic2['last_speech_time'] = current_time
            elif state_mic2['speaking']:
                state_mic2['silence'] += 1
                state_mic2['frames'].append(frame2)
                
                if state_mic2['silence'] >= SILENCE_THRESHOLD:
                    if state_mic2['speech_count'] >= MIN_SPEECH_FRAMES:
                        # Speech ended, record when it stopped
                        state_mic2['speaking'] = False
                        state_mic2['speech_end_time'] = current_time
                    # Reset state but preserve tracking info
                    last_speech_time = state_mic2.get('last_speech_time')
                    speech_end_time = state_mic2.get('speech_end_time', current_time)
                    was_speaking = state_mic2['was_speaking']
                    state_mic2 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0,
                                 'last_speech_time': last_speech_time,
                                 'speech_end_time': speech_end_time,
                                 'was_speaking': was_speaking}
            
            # Determine current status (priority: green > blue > yellow)
            new_status = None
            if state_mic1['speaking']:
                new_status = 'green'
            elif state_mic2['speaking']:
                new_status = 'blue'
            elif state_mic2.get('was_speaking') and state_mic2.get('speech_end_time'):
                # Check if other person has been silent for 2+ seconds after they stopped speaking
                silence_duration = current_time - state_mic2['speech_end_time']
                if silence_duration >= SILENCE_TIMEOUT_SECONDS:
                    new_status = 'yellow'
            
            # Print status if it changed or enough time has passed
            if new_status != current_status or (current_time - last_status_print_time >= status_print_interval):
                if new_status:
                    print_status(new_status, current_time)
                    # Update state file for UI
                    update_transition_state(new_status)
                    current_status = new_status
                    last_status_print_time = current_time
                
    except KeyboardInterrupt:
        print("\n\n=== Stopped ===")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stream_mic1.stop_stream()
        stream_mic1.close()
        stream_mic2.stop_stream()
        stream_mic2.close()
        p.terminate()
        print("✓ Streams closed")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()