#!/usr/bin/env python3
"""
Process Video Audio - Extract audio and generate transcriptions for demo rendering.

Pass 1 of the two-pass demo workflow:
1. Extract audio from video
2. Run Whisper transcription with timestamps
3. Run GPT summarization
4. Generate timed state JSON file

Then use render_demo.py --script output.json for smooth rendering.

Usage:
    python process_video_audio.py input.mp4 output_script.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import social cue detector
from audio.social_cue_detector import detect_social_cues

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def find_ffmpeg() -> str:
    """Find ffmpeg executable, checking common installation locations."""
    # Check PATH first
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    # Common Windows locations
    common_paths = [
        r"C:\Program Files\ShareX\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]

    for path in common_paths:
        if Path(path).exists():
            return path

    return None


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video as 16kHz mono WAV for Whisper."""
    print(f"[AUDIO] Extracting audio from {video_path.name}...")

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[AUDIO] Error: ffmpeg not found. Please install ffmpeg.")
        return False

    print(f"[AUDIO] Using ffmpeg: {ffmpeg}")

    try:
        cmd = [
            ffmpeg, '-y', '-i', str(video_path),
            '-vn',                    # No video
            '-acodec', 'pcm_s16le',   # PCM 16-bit
            '-ar', '16000',           # 16kHz for Whisper
            '-ac', '1',               # Mono
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[AUDIO] ffmpeg error: {result.stderr}")
            return False
        print(f"[AUDIO] Audio extracted to {audio_path}")
        return True
    except FileNotFoundError:
        print("[AUDIO] Error: ffmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        print(f"[AUDIO] Error extracting audio: {e}")
        return False


def split_long_segments(segments: list, max_duration: float = 5.0) -> list:
    """Post-process Whisper segments to split overly long ones.

    - Split segments longer than max_duration at natural pause points
    - Split segments containing multiple sentences (". " or "? ")
    - Distribute timestamps proportionally based on text length
    """
    if not segments:
        return segments

    result = []

    for seg in segments:
        duration = seg['end'] - seg['start']
        text = seg['text'].strip()

        # Check if segment needs splitting
        needs_split = False

        # Condition 1: Too long
        if duration > max_duration:
            needs_split = True

        # Condition 2: Contains multiple sentences
        # Look for sentence boundaries: ". ", "? ", "! " followed by capital letter or space
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        if len(sentences) > 1:
            needs_split = True

        if needs_split and len(sentences) > 1:
            # Split into multiple segments
            total_chars = sum(len(s) for s in sentences)
            current_time = seg['start']

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Distribute time proportionally based on character count
                char_ratio = len(sentence) / total_chars if total_chars > 0 else 1.0 / len(sentences)
                seg_duration = duration * char_ratio

                new_seg = {
                    'start': current_time,
                    'end': current_time + seg_duration,
                    'text': sentence
                }
                result.append(new_seg)
                print(f"  [SPLIT] [{new_seg['start']:.1f}s - {new_seg['end']:.1f}s] {sentence[:50]}...")

                current_time += seg_duration
        else:
            # Keep segment as-is
            result.append(seg)

    print(f"[WHISPER] After splitting: {len(result)} segments (was {len(segments)})")
    return result


def transcribe_audio(audio_path: Path) -> list:
    """Transcribe audio using OpenAI Whisper API with timestamps."""
    print("[WHISPER] Transcribing audio...")

    try:
        from openai import OpenAI
        client = OpenAI()

        with open(audio_path, 'rb') as audio_file:
            # Use whisper-1 model with timestamp granularities
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                # Handle both dict and object formats
                start = seg.start if hasattr(seg, 'start') else seg['start']
                end = seg.end if hasattr(seg, 'end') else seg['end']
                text = seg.text if hasattr(seg, 'text') else seg['text']
                text = text.strip()
                segments.append({
                    'start': start,
                    'end': end,
                    'text': text
                })
                print(f"  [{start:.1f}s - {end:.1f}s] {text}")

        print(f"[WHISPER] Transcribed {len(segments)} raw segments")

        # Post-process: split long segments
        segments = split_long_segments(segments)

        return segments

    except Exception as e:
        print(f"[WHISPER] Error: {e}")
        return []


def detect_question_regex(text: str) -> str:
    """Backup regex-based question detection for sentences ending in '?'."""
    # Split into sentences and find questions
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    questions = [s.strip() for s in sentences if s.strip().endswith('?')]
    if questions:
        # Return the last question asked
        return questions[-1]
    return ''


def generate_summaries(segments: list) -> list:
    """Generate conversation summaries using GPT for each segment.

    Implements baseball-style stacking captions:
    - recent_utterance: The current/latest utterance (NEVER empty or "...")
    - recent_utterances: List of last 2-3 utterances for stacked display
    - utterance_times: Timestamps for each utterance in the stack
    """
    print("[GPT] Generating summaries...")

    # Maximum number of utterances to keep in the stack
    MAX_UTTERANCE_STACK = 3

    try:
        from openai import OpenAI
        client = OpenAI()

        # Group segments into chunks for summarization
        states = []
        context_window = []

        # Track stacking utterances (list of {text, time} dicts)
        utterance_stack = []

        for i, seg in enumerate(segments):
            context_window.append(seg['text'])

            # Keep last 3 utterances for context
            if len(context_window) > 3:
                context_window.pop(0)

            # Add current utterance to the stack
            utterance_stack.append({
                'text': seg['text'],
                'time': seg['end']
            })

            # Keep only last MAX_UTTERANCE_STACK utterances
            if len(utterance_stack) > MAX_UTTERANCE_STACK:
                utterance_stack.pop(0)

            # Generate summary
            context_text = "\n".join([f"- {t}" for t in context_window])

            prompt = f"""Analyze this conversation segment for an AR display assistant.

Recent conversation:
{context_text}

Current utterance to analyze: "{seg['text']}"

Your task:
1. SUMMARY: Provide a brief summary of what's being discussed (max 60 characters)
2. QUESTION DETECTION: Carefully check if the speaker asked ANY question in the current utterance.
   - Look for direct questions (ending in ?)
   - Look for indirect questions ("I wonder if...", "Do you know...", "Can you tell me...")
   - Look for requests phrased as questions ("Could you...", "Would you mind...")
   - Look for tag questions ("...right?", "...isn't it?", "...don't you think?")
   If a question was asked, extract the FULL question. If no question, use empty string.
3. VIBE: Assess the speaker's engagement level (engaged/bored/upset/excited/neutral)

Respond with ONLY a valid JSON object:
{{"summary": "<max 60 chars>", "question": "<the full question if asked, otherwise empty string>", "vibe": "<one word>"}}"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing conversations for an AR assistant that helps neurodivergent users. You are especially skilled at detecting questions - both explicit (ending in ?) and implicit (requests, indirect questions). Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )

                result_text = response.choices[0].message.content.strip()
                # Clean up markdown if present
                if result_text.startswith('```'):
                    result_text = result_text.split('\n', 1)[1].rsplit('\n', 1)[0]

                result = json.loads(result_text)

                # Use GPT question detection, fall back to regex if empty
                detected_question = result.get('question', '')
                if not detected_question:
                    detected_question = detect_question_regex(seg['text'])

                # Detect social cues
                social_cues = detect_social_cues(seg['text'], emotion=None)

                # Build stacked utterances list (just the text strings)
                recent_utterances = [u['text'] for u in utterance_stack]
                utterance_times = [u['time'] for u in utterance_stack]

                state_data = {
                    'convo_state_summary': result.get('summary', 'Listening...'),
                    # recent_utterance is ALWAYS the current utterance (never empty)
                    'recent_utterance': seg['text'],
                    # Stacked utterances for baseball-style display
                    'recent_utterances': recent_utterances,
                    'utterance_times': utterance_times,
                    'question': detected_question,
                    'emotion': 'neutral',  # Will be filled by emotion detection during render
                    'emotion_display': 'Neutral',
                    'is_other_speaking': True,
                    'vibe': result.get('vibe', 'neutral'),
                    'segment_start': seg['start'],
                    'segment_end': seg['end']
                }

                # Add social cue info if detected
                if social_cues:
                    # Use the first/most important social cue
                    cue = social_cues[0]
                    state_data['social_cue'] = cue.message
                    state_data['social_cue_icon'] = cue.icon
                    state_data['social_cue_timestamp'] = seg['end']
                    print(f"  [SOCIAL CUE] {cue.icon} {cue.message}")

                states.append({
                    'time': seg['end'],  # Show after speech finishes
                    'state': state_data
                })

                print(f"  [{seg['start']:.1f}s] {result.get('summary', '')[:50]}...")
                print(f"    [STACK] {len(recent_utterances)} utterances in stack")
                if detected_question:
                    print(f"    [QUESTION] {detected_question[:60]}...")

            except json.JSONDecodeError:
                # Fallback if GPT doesn't return valid JSON
                detected_question = detect_question_regex(seg['text'])
                social_cues = detect_social_cues(seg['text'], emotion=None)

                # Build stacked utterances list
                recent_utterances = [u['text'] for u in utterance_stack]
                utterance_times = [u['time'] for u in utterance_stack]

                state_data = {
                    'convo_state_summary': 'Listening...',
                    'recent_utterance': seg['text'],
                    'recent_utterances': recent_utterances,
                    'utterance_times': utterance_times,
                    'question': detected_question,
                    'emotion': 'neutral',
                    'emotion_display': 'Neutral',
                    'is_other_speaking': True,
                    'segment_start': seg['start'],
                    'segment_end': seg['end']
                }

                if social_cues:
                    cue = social_cues[0]
                    state_data['social_cue'] = cue.message
                    state_data['social_cue_icon'] = cue.icon
                    state_data['social_cue_timestamp'] = seg['end']

                states.append({
                    'time': seg['end'],  # Show after speech finishes
                    'state': state_data
                })

        print(f"[GPT] Generated {len(states)} state entries")
        return states

    except Exception as e:
        print(f"[GPT] Error: {e}")
        # Return basic states without summaries
        result_states = []
        utterance_stack = []

        for seg in segments:
            detected_question = detect_question_regex(seg['text'])
            social_cues = detect_social_cues(seg['text'], emotion=None)

            # Add to stack
            utterance_stack.append({
                'text': seg['text'],
                'time': seg['end']
            })
            if len(utterance_stack) > MAX_UTTERANCE_STACK:
                utterance_stack.pop(0)

            recent_utterances = [u['text'] for u in utterance_stack]
            utterance_times = [u['time'] for u in utterance_stack]

            state_data = {
                'convo_state_summary': 'Listening...',
                'recent_utterance': seg['text'],
                'recent_utterances': recent_utterances,
                'utterance_times': utterance_times,
                'question': detected_question,
                'emotion': 'neutral',
                'emotion_display': 'Neutral',
                'is_other_speaking': True,
                'segment_start': seg['start'],
                'segment_end': seg['end']
            }

            if social_cues:
                cue = social_cues[0]
                state_data['social_cue'] = cue.message
                state_data['social_cue_icon'] = cue.icon
                state_data['social_cue_timestamp'] = seg['end']

            result_states.append({
                'time': seg['end'],
                'state': state_data
            })
        return result_states


def add_speaking_indicators(states: list) -> list:
    """Add speaking/pause indicators between segments.

    Timing logic:
    - is_other_speaking = True during speech (from segment_start to segment_end)
    - is_other_speaking = False after speech ends (at segment_end)
    - If gap > 1.5 seconds between segments, add explicit pause state

    IMPORTANT: Baseball-style stacking captions:
    - recent_utterance is NEVER empty or "..." - always shows last actual utterance
    - recent_utterances list is inherited by pause states
    - During speech start, show PREVIOUS utterances (not current one being spoken)
    """
    if not states:
        return states

    enhanced = []

    # Track the last known utterance info for inheritance
    # This ensures recent_utterance is NEVER empty
    last_utterance = ''
    last_utterances = []
    last_utterance_times = []

    # Add initial state - will be updated once we have first utterance
    initial_state = {
        'time': 0.0,
        'state': {
            'convo_state_summary': 'Starting...',
            'recent_utterance': '',  # Will be empty only at very start
            'recent_utterances': [],
            'utterance_times': [],
            'question': '',
            'emotion': 'neutral',
            'emotion_display': 'Neutral',
            'is_other_speaking': False
        }
    }
    enhanced.append(initial_state)

    for i, state in enumerate(states):
        # Get segment timing from state data
        seg_start = state['state'].get('segment_start', state['time'] - 2.0)
        seg_end = state['state'].get('segment_end', state['time'])

        # Get current utterance info from this state
        current_utterance = state['state'].get('recent_utterance', '')
        current_utterances = state['state'].get('recent_utterances', [current_utterance] if current_utterance else [])
        current_utterance_times = state['state'].get('utterance_times', [seg_end] if current_utterance else [])

        # Add state when speech STARTS (is_other_speaking = True)
        # During speech start, show PREVIOUS utterances (what was said before)
        # This way we don't show the current utterance until it's finished
        speaking_state = {
            'time': seg_start,
            'state': {
                'convo_state_summary': state['state'].get('convo_state_summary', 'Listening...'),
                # Show PREVIOUS utterance during speech (never empty after first utterance)
                'recent_utterance': last_utterance if last_utterance else current_utterance,
                'recent_utterances': last_utterances if last_utterances else current_utterances,
                'utterance_times': last_utterance_times if last_utterance_times else current_utterance_times,
                'question': '',  # Don't show question until speech ends
                'emotion': state['state'].get('emotion', 'neutral'),
                'emotion_display': state['state'].get('emotion_display', 'Neutral'),
                'is_other_speaking': True,
                'vibe': state['state'].get('vibe', 'neutral')
            }
        }
        # Copy social cue if present (show during speech)
        if 'social_cue' in state['state']:
            speaking_state['state']['social_cue'] = state['state']['social_cue']
            speaking_state['state']['social_cue_icon'] = state['state']['social_cue_icon']
            speaking_state['state']['social_cue_timestamp'] = state['state']['social_cue_timestamp']
        enhanced.append(speaking_state)

        # Add state when speech ENDS (show NEW utterance with stacked history)
        end_state = {
            'time': seg_end,
            'state': {
                'convo_state_summary': state['state'].get('convo_state_summary', 'Listening...'),
                # Now show the CURRENT utterance (what was just said)
                'recent_utterance': current_utterance,
                'recent_utterances': current_utterances,
                'utterance_times': current_utterance_times,
                'question': state['state'].get('question', ''),
                'emotion': state['state'].get('emotion', 'neutral'),
                'emotion_display': state['state'].get('emotion_display', 'Neutral'),
                'is_other_speaking': True,  # Still speaking at the moment speech ends
                'vibe': state['state'].get('vibe', 'neutral')
            }
        }
        if 'social_cue' in state['state']:
            end_state['state']['social_cue'] = state['state']['social_cue']
            end_state['state']['social_cue_icon'] = state['state']['social_cue_icon']
            end_state['state']['social_cue_timestamp'] = state['state']['social_cue_timestamp']
        enhanced.append(end_state)

        # Update last known utterance info for next iteration
        # This is what will be shown during the NEXT speech segment
        if current_utterance:
            last_utterance = current_utterance
        if current_utterances:
            last_utterances = current_utterances.copy()
        if current_utterance_times:
            last_utterance_times = current_utterance_times.copy()

        # Determine if there's a pause after this segment
        if i < len(states) - 1:
            # Get next segment's start time
            next_seg_start = states[i + 1]['state'].get('segment_start', states[i + 1]['time'] - 2.0)
            gap = next_seg_start - seg_end
        else:
            # After last segment, always add a pause state
            gap = 2.0  # Force pause after last segment

        # Add pause state if gap > 1.5 seconds
        if gap > 1.5:
            pause_time = seg_end + 0.1  # Small delay after speech ends
            pause_state = {
                'time': pause_time,
                'state': {
                    'convo_state_summary': state['state'].get('convo_state_summary', 'Listening...'),
                    # INHERIT utterances during pause - never clear them!
                    'recent_utterance': current_utterance if current_utterance else last_utterance,
                    'recent_utterances': current_utterances if current_utterances else last_utterances,
                    'utterance_times': current_utterance_times if current_utterance_times else last_utterance_times,
                    'question': state['state'].get('question', ''),
                    'emotion': state['state'].get('emotion', 'neutral'),
                    'emotion_display': state['state'].get('emotion_display', 'Neutral'),
                    'is_other_speaking': False,  # Other person stopped speaking
                    'vibe': state['state'].get('vibe', 'neutral')
                }
            }
            # Keep social cue visible during pause
            if 'social_cue' in state['state']:
                pause_state['state']['social_cue'] = state['state']['social_cue']
                pause_state['state']['social_cue_icon'] = state['state']['social_cue_icon']
                pause_state['state']['social_cue_timestamp'] = state['state']['social_cue_timestamp']
            enhanced.append(pause_state)
            print(f"  [PAUSE] Added pause state at {pause_time:.1f}s (gap: {gap:.1f}s)")

    # Sort by time to ensure correct ordering
    enhanced.sort(key=lambda x: x['time'])

    # Post-process: ensure no state has empty recent_utterance after first real utterance
    # Find the first non-empty utterance
    first_utterance = ''
    first_utterances = []
    first_utterance_times = []
    for state in enhanced:
        if state['state'].get('recent_utterance'):
            first_utterance = state['state']['recent_utterance']
            first_utterances = state['state'].get('recent_utterances', [first_utterance])
            first_utterance_times = state['state'].get('utterance_times', [])
            break

    # Fill in any empty utterances with the first available (for initial states)
    if first_utterance:
        for state in enhanced:
            if not state['state'].get('recent_utterance'):
                # Only fill if this is before the first utterance time
                state['state']['recent_utterance'] = first_utterance
                state['state']['recent_utterances'] = first_utterances
                state['state']['utterance_times'] = first_utterance_times

    return enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Process video audio for demo rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two-pass demo workflow:
  1. python process_video_audio.py input.mp4 script.json
  2. python render_demo.py input.mp4 output.mp4 --script script.json
        """
    )

    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output JSON script file')
    parser.add_argument('--keep-audio', action='store_true',
                        help='Keep extracted audio file')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Video Audio Processor")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Extract audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        if not extract_audio(input_path, audio_path):
            sys.exit(1)

        # Transcribe
        segments = transcribe_audio(audio_path)
        if not segments:
            print("Warning: No speech detected in audio")

        # Generate summaries
        states = generate_summaries(segments)

        # Add speaking indicators
        states = add_speaking_indicators(states)

        # Save script
        with open(output_path, 'w') as f:
            json.dump(states, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Script saved to: {output_path}")
        print(f"Total state changes: {len(states)}")
        print(f"\nNext step:")
        print(f"  python render_demo.py {input_path} output.mp4 --script {output_path}")
        print(f"{'='*60}\n")

    finally:
        # Cleanup
        if not args.keep_audio and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()
