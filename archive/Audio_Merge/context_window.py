import pyaudio
import webrtcvad
import wave
import tempfile
import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
from queue import Queue, Empty
import threading
from pathlib import Path
import webview

load_dotenv()
client = OpenAI()

# Audio settings
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# VAD settings
VAD_AGGRESSIVENESS = 3
SILENCE_THRESHOLD = 20
MIN_SPEECH_FRAMES = 8

# Processing settings
MAX_UTTERANCES_BEFORE_UPDATE = 3
MAX_WAIT_TIME = 3.0
MIN_WORDS_FOR_UPDATE = 15

# Create directories
USER_DIR = Path("user_conv_context")
PERSON_DIR = Path("person_talking_to_context")
USER_DIR.mkdir(exist_ok=True)
PERSON_DIR.mkdir(exist_ok=True)

# Conversation helper UI HTML
CONVO_HELPER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Helper</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
            width: 100%;
            overflow: hidden;
            background: transparent;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            color: white;
            -webkit-font-smoothing: antialiased;
        }

        .main-container {
            position: absolute;
            top: 8px;
            left: 8px;
            right: 8px;
            bottom: 8px;
            background: rgba(35, 35, 40, 0.92);
            backdrop-filter: blur(80px) saturate(150%);
            -webkit-backdrop-filter: blur(80px) saturate(150%);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow:
                0 25px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .content {
            flex: 1;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow: hidden;
        }

        ::-webkit-scrollbar {
            display: none;
        }

        * {
            scrollbar-width: none;
            -ms-overflow-style: none;
        }

        .context-section {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 10px 12px;
            flex-shrink: 0;
            overflow: hidden;
        }

        .section-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255, 255, 255, 0.35);
            margin-bottom: 4px;
        }

        .main-topic {
            font-size: 15px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.95);
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .sub-context {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.95);
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .question-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 12px;
            padding: 10px 12px;
            flex-shrink: 0;
            overflow: hidden;
        }

        .question-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(167, 139, 250, 0.7);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .question-icon {
            width: 12px;
            height: 12px;
            flex-shrink: 0;
        }

        .question-text {
            font-size: 13px;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.95);
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .no-question {
            color: rgba(255, 255, 255, 0.3);
            font-style: italic;
        }

        .utterance-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 8px 10px;
            flex-shrink: 0;
            overflow: hidden;
        }

        .utterance-text {
            font-size: 11px;
            color: rgba(255, 255, 255, 0.95);
            line-height: 1.35;
            font-style: italic;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="content">
            <div class="context-section">
                <div class="section-label">Conversation Summary</div>
                <div class="main-topic" id="conversation-summary">Waiting for conversation...</div>
            </div>

            <div class="utterance-box">
                <div class="section-label">Last Updated</div>
                <div class="utterance-text" id="timestamp">--</div>
            </div>
        </div>
    </div>

    <script>
        function updateUI(data) {
            const summaryEl = document.getElementById('conversation-summary');
            const timestampEl = document.getElementById('timestamp');
            
            if (data.conversation_summary) {
                summaryEl.textContent = data.conversation_summary;
                summaryEl.classList.add('fade-in');
                setTimeout(() => summaryEl.classList.remove('fade-in'), 300);
            }
            
            if (data.timestamp) {
                timestampEl.textContent = data.timestamp;
            }
        }

        // Poll for updates from Python
        if (window.pywebview) {
            setInterval(async () => {
                try {
                    const data = await pywebview.api.get_final_state();
                    if (data) {
                        updateUI(data);
                    }
                } catch (e) {
                    console.error('Error fetching state:', e);
                }
            }, 1000);
        }
    </script>
</body>
</html>
"""


class ConversationAPI:
    """API exposed to JavaScript for accessing conversation state."""

    def __init__(self):
        self.final_state_path = PERSON_DIR / "final_state.json"

    def get_final_state(self):
        """Get current final_state.json contents."""
        try:
            if self.final_state_path.exists():
                with open(self.final_state_path, 'r') as f:
                    return json.load(f)
            return {
                "conversation_summary": "No conversation data yet.",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            return {
                "conversation_summary": f"Error loading state: {str(e)}",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def close_window(self):
        """Close the window (called from JS)."""
        pass  # Window will be destroyed externally


def transcribe_audio(audio_bytes):
    """Transcribe audio using OpenAI Whisper"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_bytes)
        
        try:
            with open(tmp_file.name, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return transcript.text.strip()
        except Exception as e:
            print(f"   ⚠️  Transcription error: {e}")
            return ""
        finally:
            os.unlink(tmp_file.name)


def get_next_context_number(directory):
    """Get next context file number (cycles 1-5)"""
    existing = list(directory.glob("context_*.json"))
    
    if not existing:
        return 1
    
    nums = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[1])
            if 1 <= num <= 5:
                nums.append(num)
        except:
            f.unlink()
    
    if not nums:
        return 1
    
    file_ages = [(directory / f"context_{n:03d}.json", n) for n in nums]
    file_ages = [(f, n) for f, n in file_ages if f.exists()]
    
    if len(nums) < 5:
        for i in range(1, 6):
            if i not in nums:
                return i
    
    if file_ages:
        oldest = min(file_ages, key=lambda x: x[0].stat().st_mtime)
        return oldest[1]
    
    return 1


def get_most_recent_context_files(directory, count=2):
    """Get the most recent N context files from a directory"""
    existing = list(directory.glob("context_*.json"))
    
    if not existing:
        return []
    
    existing.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    recent_files = []
    for f in existing[:count]:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                recent_files.append({
                    'filename': f.name,
                    'number': int(f.stem.split('_')[1]),
                    'data': data
                })
        except:
            continue
    
    return recent_files


def save_context(directory, text, utterance_count, word_count):
    """Save context to numbered JSON file (1-5)"""
    num = get_next_context_number(directory)
    filename = directory / f"context_{num:03d}.json"
    
    data = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "text": text,
        "utterance_count": utterance_count,
        "word_count": word_count
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filename


def generate_conversation_summary():
    """Generate final_state summary using GPT based on recent context from both folders"""
    user_contexts = get_most_recent_context_files(USER_DIR, count=2)
    person_contexts = get_most_recent_context_files(PERSON_DIR, count=2)
    
    if not user_contexts and not person_contexts:
        return {
            "conversation_summary": "No conversation data available yet.",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    user_text = ""
    if user_contexts:
        user_text = "USER's recent statements:\n"
        for ctx in sorted(user_contexts, key=lambda x: x['data']['timestamp']):
            user_text += f"- [{ctx['data']['timestamp']}] {ctx['data']['text']}\n"
    
    person_text = ""
    if person_contexts:
        person_text = "\nPERSON TALKING TO's recent statements:\n"
        for ctx in sorted(person_contexts, key=lambda x: x['data']['timestamp']):
            person_text += f"- [{ctx['data']['timestamp']}] {ctx['data']['text']}\n"
    
    prompt = f"""Based on the following conversation snippets, provide a concise summary of the current conversation state - what they're discussing and the key points so far.

{user_text}{person_text}

Provide a brief summary (2-3 sentences max) of what the conversation is about and its current state."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a conversation analyst. Summarize conversation states concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        summary = response.choices[0].message.content.strip()
        
        return {
            "conversation_summary": summary,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "user_context_files_used": [ctx['filename'] for ctx in user_contexts],
            "person_context_files_used": [ctx['filename'] for ctx in person_contexts]
        }
    except Exception as e:
        print(f"   ⚠️  Summary generation error: {e}")
        return {
            "conversation_summary": f"Error generating summary: {str(e)}",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }


def update_final_state():
    """Update final_state.json with summary based on recent contexts"""
    final_state_path = PERSON_DIR / "final_state.json"
    
    summary_data = generate_conversation_summary()
    
    with open(final_state_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    return final_state_path


def process_utterances(utterances_batch, mic_label, directory):
    """Process batch of utterances"""
    if not utterances_batch:
        return
    
    combined_text = " ".join(utterances_batch)
    word_count = len(combined_text.split())
    
    print(f"\n[{time.strftime('%H:%M:%S')}] {mic_label} \"{combined_text}\"")
    
    context_file = save_context(directory, combined_text, len(utterances_batch), word_count)
    print(f"   💾 {context_file.name}")
    
    try:
        update_final_state()
        print(f"   🔄 final_state.json updated")
    except Exception as e:
        print(f"   ⚠️  Failed to update final_state: {e}")


def transcription_processor(audio_queue, mic_label, directory):
    """Background processor for transcriptions"""
    utterances_batch = []
    last_utterance_time = None
    
    while True:
        try:
            utterance_data = audio_queue.get(timeout=0.1)
            text = transcribe_audio(utterance_data['audio'])
            
            if text:
                utterances_batch.append(text)
                last_utterance_time = time.time()
                word_count = sum(len(u.split()) for u in utterances_batch)
                
                should_process = (
                    len(utterances_batch) >= MAX_UTTERANCES_BEFORE_UPDATE or
                    word_count >= MIN_WORDS_FOR_UPDATE
                )
                
                if should_process:
                    process_utterances(utterances_batch, mic_label, directory)
                    utterances_batch = []
                    last_utterance_time = None
                    
        except Empty:
            if utterances_batch and last_utterance_time:
                if time.time() - last_utterance_time >= MAX_WAIT_TIME:
                    process_utterances(utterances_batch, mic_label, directory)
                    utterances_batch = []
                    last_utterance_time = None
        except Exception as e:
            print(f"{mic_label} Error: {e}")
            break


def start_webview():
    """Start the conversation helper webview in a separate thread"""
    api = ConversationAPI()
    
    window = webview.create_window(
        'Conversation Helper',
        html=CONVO_HELPER_HTML,
        js_api=api,
        width=320,
        height=360,
        resizable=True,
        frameless=True,
        easy_drag=True,
        transparent=True,
        background_color='#0a0a0f',
    )
    
    webview.start(debug=False)


def main():
    # Start webview in separate thread
    webview_thread = threading.Thread(target=start_webview, daemon=True)
    webview_thread.start()
    
    # Give webview time to initialize
    time.sleep(2)
    
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
    print(f"Mic 1: [{mic1_index}] {mic1_name} → {USER_DIR}/")
    print(f"Mic 2: [{mic2_index}] {mic2_name} → {PERSON_DIR}/")
    print("="*60 + "\n")

    # Setup queues and threads
    audio_queue_mic1 = Queue()
    audio_queue_mic2 = Queue()

    threading.Thread(target=transcription_processor, args=(audio_queue_mic1, "[MIC1-USER]", USER_DIR), daemon=True).start()
    threading.Thread(target=transcription_processor, args=(audio_queue_mic2, "[MIC2-OTHER]", PERSON_DIR), daemon=True).start()

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
    state_mic2 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}

    print("🎤 Listening... (Press Ctrl+C to stop)\n")

    try:
        while True:
            # Process Mic 1
            frame1 = stream_mic1.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech1 = vad_mic1.is_speech(frame1, SAMPLE_RATE)
            
            if is_speech1:
                if not state_mic1['speaking']:
                    print("\n🗣️  [MIC1-USER] Speech")
                    state_mic1['speaking'] = True
                state_mic1['frames'].append(frame1)
                state_mic1['speech_count'] += 1
                state_mic1['silence'] = 0
            elif state_mic1['speaking']:
                state_mic1['silence'] += 1
                state_mic1['frames'].append(frame1)
                
                if state_mic1['silence'] >= SILENCE_THRESHOLD:
                    if state_mic1['speech_count'] >= MIN_SPEECH_FRAMES:
                        duration_ms = len(state_mic1['frames']) * FRAME_DURATION_MS
                        print(f"   ✓ Complete ({duration_ms}ms)")
                        audio_queue_mic1.put({'audio': b''.join(state_mic1['frames']), 'timestamp': time.time()})
                    state_mic1 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}
            
            # Process Mic 2
            frame2 = stream_mic2.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech2 = vad_mic2.is_speech(frame2, SAMPLE_RATE)
            
            if is_speech2:
                if not state_mic2['speaking']:
                    print("\n🗣️  [MIC2-OTHER] Speech")
                    state_mic2['speaking'] = True
                state_mic2['frames'].append(frame2)
                state_mic2['speech_count'] += 1
                state_mic2['silence'] = 0
            elif state_mic2['speaking']:
                state_mic2['silence'] += 1
                state_mic2['frames'].append(frame2)
                
                if state_mic2['silence'] >= SILENCE_THRESHOLD:
                    if state_mic2['speech_count'] >= MIN_SPEECH_FRAMES:
                        duration_ms = len(state_mic2['frames']) * FRAME_DURATION_MS
                        print(f"   ✓ Complete ({duration_ms}ms)")
                        audio_queue_mic2.put({'audio': b''.join(state_mic2['frames']), 'timestamp': time.time()})
                    state_mic2 = {'frames': [], 'silence': 0, 'speaking': False, 'speech_count': 0}
                
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


if __name__ == "__main__":
    main()
