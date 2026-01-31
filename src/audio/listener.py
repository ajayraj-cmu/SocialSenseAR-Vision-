"""Wake word voice listener - say 'hey vibe' for video, 'hey vibe audio' for audio."""
import threading
import queue
import time
import speech_recognition as sr


class VoiceListener:
    """Wake word voice listener - say 'hey vibe' for video, 'hey vibe audio' for audio."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.audio_command_queue = queue.Queue()
        self.listening = True
        self.is_recording = False
        self.is_processing = False
        self.recording_mode = None
        self.last_command = ""
        self.status = "👂 Listening for 'hey vibe' or 'hey vibe audio'..."
        self._lock = threading.Lock()

        self.video_wake_phrases = ["hey vibe"]
        self.audio_wake_phrases = ["hey vibe audio", "hey vibe audio mode", "audio mode"]
        self.stop_phrases = ["thanks", "thank you", "done", "stop"]

        with self.microphone as source:
            print("🎤 Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        print("✅ Voice listener ready:")
        print("   • Say 'hey vibe' for video commands")
        print("   • Say 'hey vibe audio' for audio commands")
        print("   • Say 'thanks' to process")

        self._listening_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self._listening_thread.start()

    def _continuous_listen(self):
        """Continuously listen for wake words: 'hey vibe' (video) or 'hey vibe audio'."""
        while self.listening:
            if self.is_recording or self.is_processing:
                time.sleep(0.5)
                continue

            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)

                try:
                    text = self.recognizer.recognize_google(audio).lower()

                    if any(phrase in text for phrase in self.audio_wake_phrases):
                        with self._lock:
                            if not self.is_recording and not self.is_processing:
                                self.is_recording = True
                                self.recording_mode = "audio"
                            else:
                                continue

                        self.status = "🎵 AUDIO MODE - say your audio command, then 'thanks'"
                        print("🎵 Audio wake word detected! Listening for audio command...")
                        time.sleep(0.3)
                        threading.Thread(target=self._record_command, daemon=True).start()

                    elif any(phrase in text for phrase in self.video_wake_phrases):
                        with self._lock:
                            if not self.is_recording and not self.is_processing:
                                self.is_recording = True
                                self.recording_mode = "video"
                            else:
                                continue

                        self.status = "🔴 RECORDING... say your command, then 'thanks'"
                        print("🎤 Video wake word detected! Listening for command...")
                        time.sleep(0.3)
                        threading.Thread(target=self._record_command, daemon=True).start()

                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    if self.listening:
                        time.sleep(0.5)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                if self.listening:
                    print(f"⚠️ Listening error: {e}")
                    time.sleep(0.5)

    def _record_command(self):
        """Record command until 'thanks' is heard."""
        command_parts = []
        max_phrases = 10
        phrase_count = 0

        cmd_recognizer = sr.Recognizer()
        cmd_recognizer.energy_threshold = 300
        cmd_recognizer.pause_threshold = 0.8

        while self.is_recording and phrase_count < max_phrases:
            try:
                with self.microphone as source:
                    cmd_recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = cmd_recognizer.listen(source, timeout=5, phrase_time_limit=8)

                try:
                    text = cmd_recognizer.recognize_google(audio).lower()

                    if any(phrase in text for phrase in self.stop_phrases):
                        for phrase in self.stop_phrases:
                            text = text.replace(phrase, "").strip()
                        if text:
                            command_parts.append(text)

                        with self._lock:
                            self.is_recording = False
                        self.status = "🔄 Processing..."
                        break
                    else:
                        command_parts.append(text)
                        phrase_count += 1
                        self.status = f"🔴 Recording... ({phrase_count}/{max_phrases}) say 'thanks' when done"

                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"⚠️ Speech API error: {e}")
                    with self._lock:
                        self.is_recording = False
                    break

            except sr.WaitTimeoutError:
                if command_parts:
                    with self._lock:
                        self.is_recording = False
                    break
                continue
            except OSError as e:
                print(f"⚠️ Microphone error: {e}")
                with self._lock:
                    self.is_recording = False
                break
            except Exception as e:
                print(f"⚠️ Recording error: {e}")
                with self._lock:
                    self.is_recording = False
                break

        if command_parts:
            full_command = " ".join(command_parts).strip()
            if full_command and len(full_command) > 2:
                with self._lock:
                    self.is_processing = True

                self.last_command = full_command

                if self.recording_mode == "audio":
                    self.audio_command_queue.put(full_command)
                    self.status = f"🎵 Audio: {full_command[:40]}..."
                    print(f"🎵 Audio command: {full_command}")
                else:
                    self.command_queue.put(full_command)
                    self.status = f"✅ Got: {full_command[:40]}..."
                    print(f"🗣️ Video command: {full_command}")
            else:
                self.status = "👂 Listening for 'hey vibe' or 'hey vibe audio'..."
        else:
            self.status = "👂 Listening for 'hey vibe' or 'hey vibe audio'..."

        with self._lock:
            self.is_recording = False
            self.is_processing = False
            self.recording_mode = None

    def is_busy(self):
        """Check if currently recording or processing."""
        with self._lock:
            return self.is_recording or self.is_processing

    def get_command(self):
        """Get next video command from queue (non-blocking)."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def get_audio_command(self):
        """Get next audio command from queue (non-blocking)."""
        try:
            return self.audio_command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.listening = False
