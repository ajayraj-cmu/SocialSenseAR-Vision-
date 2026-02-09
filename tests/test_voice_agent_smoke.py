"""Smoke test for voice agent pipeline — verifies basic integration without running full server.

Tests:
1. VoiceAgent initialization
2. Wake word detection
3. Utterance assembly
4. Command planning (with/without Gemini)
5. Effect application to segments
6. Protobuf serialization

Run:
    python test_voice_agent_smoke.py
"""

import sys
import os
sys.path.insert(0, ".")

from server.audio.voice_agent import WakeWordGate, UtteranceAssembler, VoiceCommandPlanner, VoiceAgent
from server.vision.segment_data import SegmentData, EffectData
from server.config import ServerConfig
import numpy as np


def test_wake_word_gate():
    """Test wake word detection."""
    print("TEST: WakeWordGate")
    gate = WakeWordGate()

    # Positive cases
    assert gate.detect("Hey Vibe"), "Failed: 'Hey Vibe'"
    assert gate.detect("hey vibe"), "Failed: lowercase"
    assert gate.detect("Hey Vibes"), "Failed: 'Hey Vibes'"
    assert gate.detect("Hey Vive, blur the laptop"), "Failed: 'Hey Vive'"
    assert gate.detect("  Hey Vibe  "), "Failed: whitespace"

    # Negative cases
    assert not gate.detect("Hello there"), "False positive: 'Hello there'"
    assert not gate.detect("Hey there"), "False positive: 'Hey there'"

    # Strip wake word
    stripped = gate.strip_wake_word("Hey Vibe, blur the laptop")
    assert "Hey Vibe" not in stripped, f"Wake word not stripped: {stripped}"
    assert "blur" in stripped, f"Command lost: {stripped}"

    print("  ✓ Wake word detection works")


def test_utterance_assembler():
    """Test utterance assembly."""
    print("TEST: UtteranceAssembler")
    assembler = UtteranceAssembler()

    # Start recording
    assembler.start()
    assert assembler.is_active, "Assembler not active after start()"

    # Add chunks
    result = assembler.add_chunk("blur the laptop")
    assert result is None, "Premature completion without end phrase"

    result = assembler.add_chunk("and the monitor")
    assert result is None, "Premature completion"

    # End phrase triggers completion
    result = assembler.add_chunk("thank you")
    assert result is not None, "End phrase not detected"
    assert "blur" in result, f"Command lost: {result}"
    assert "thank you" not in result.lower(), f"End phrase not stripped: {result}"
    assert not assembler.is_active, "Assembler still active after completion"

    print("  ✓ Utterance assembly works")


def test_command_planner_fallback():
    """Test command planner fallback (no API keys)."""
    print("TEST: VoiceCommandPlanner (fallback mode)")
    planner = VoiceCommandPlanner()
    # Don't initialize (no API key) — should use fallback

    known_objects = {"laptop", "monitor", "desk"}
    active_effects = {}

    # Test explicit request
    plan = planner.plan_command("blur the laptop", known_objects, active_effects)
    assert "laptop" in plan.targets, f"Target not found: {plan.targets}"
    assert plan.effect_type == "blur", f"Wrong effect: {plan.effect_type}"
    assert plan.action == "add", f"Wrong action: {plan.action}"

    # Test removal
    plan = planner.plan_command("stop blurring the monitor", known_objects, {"monitor": {"type": "blur"}})
    assert plan.action == "remove", f"Wrong action for removal: {plan.action}"

    # Test dim effect
    plan = planner.plan_command("dim the desk", known_objects, active_effects)
    assert plan.effect_type == "dim", f"Wrong effect for dim: {plan.effect_type}"

    print("  ✓ Command planner fallback works")


def test_effect_data():
    """Test EffectData model."""
    print("TEST: EffectData")
    effect = EffectData(effect_type="blur", intensity=0.8, color_hex="#FF0000")
    assert effect.effect_type == "blur"
    assert effect.intensity == 0.8
    assert effect.color_hex == "#FF0000"
    assert isinstance(effect.params, dict)

    print("  ✓ EffectData model works")


def test_segment_data_with_effect():
    """Test SegmentData with effect."""
    print("TEST: SegmentData + EffectData")
    mask = np.zeros((100, 100), dtype=np.uint8)
    seg = SegmentData(mask, label="laptop", confidence=0.95)

    # Apply effect
    seg.effect = EffectData(effect_type="blur", intensity=0.7)
    assert seg.effect is not None
    assert seg.effect.effect_type == "blur"
    assert seg.effect.intensity == 0.7

    print("  ✓ SegmentData with effects works")


def test_voice_agent_creation():
    """Test VoiceAgent instantiation (without full initialization)."""
    print("TEST: VoiceAgent creation")

    # Create mock components
    class MockTranscriber:
        def __init__(self):
            self.available = False
        def initialize(self):
            pass
        def shutdown(self):
            pass
        def transcribe(self, audio, sr):
            return ""

    class MockSceneUnderstanding:
        def __init__(self):
            self.available = False
        def initialize(self):
            pass
        def shutdown(self):
            pass

    planner = VoiceCommandPlanner()  # fallback mode
    transcriber = MockTranscriber()
    scene = MockSceneUnderstanding()
    config = ServerConfig()

    # Create voice agent
    voice_agent = VoiceAgent(transcriber, planner, scene, config)

    # Check initial state
    assert len(voice_agent.get_known_objects()) == 0, "Known objects should start empty"
    assert len(voice_agent.get_active_effects()) == 0, "Active effects should start empty"
    assert voice_agent.get_full_screen_filter() is None, "Full screen filter should start None"

    # Add known object
    voice_agent.add_known_object("laptop")
    assert "laptop" in voice_agent.get_known_objects(), "Failed to add known object"

    print("  ✓ VoiceAgent creation works")


def test_protobuf_effect_fields():
    """Test that protobuf schema includes effect fields."""
    print("TEST: Protobuf effect fields")
    from server.proto import socialsense_pb2 as pb

    # Create message with effect
    msg = pb.ServerMessage()
    seg = msg.segments.add()
    seg.label = "laptop"
    seg.effect.effect_type = "blur"
    seg.effect.intensity = 0.8
    seg.effect.color_hex = "#FF0000"
    seg.effect.params["test"] = "value"

    # Serialize and deserialize
    data = msg.SerializeToString()
    msg2 = pb.ServerMessage()
    msg2.ParseFromString(data)

    assert msg2.segments[0].effect.effect_type == "blur"
    assert abs(msg2.segments[0].effect.intensity - 0.8) < 0.01, f"Intensity mismatch: {msg2.segments[0].effect.intensity}"
    assert msg2.segments[0].effect.color_hex == "#FF0000"
    assert msg2.segments[0].effect.params["test"] == "value"

    print("  ✓ Protobuf effect serialization works")


def test_voice_agent_state_protobuf():
    """Test voice agent state in protobuf."""
    print("TEST: VoiceAgentState protobuf")
    from server.proto import socialsense_pb2 as pb

    msg = pb.ServerMessage()
    msg.conversation.voice_agent.listening = True
    msg.conversation.voice_agent.recording = False
    msg.conversation.voice_agent.partial_transcript = "blur the"
    msg.conversation.voice_agent.last_command = "blur the laptop"
    msg.conversation.voice_agent.last_response = "Applied blur to laptop"
    msg.conversation.voice_agent.last_command_time = 12345.6

    msg.conversation.voice_agent.full_screen_filter.filter_type = "dim"
    msg.conversation.voice_agent.full_screen_filter.intensity = 0.5

    # Serialize and deserialize
    data = msg.SerializeToString()
    msg2 = pb.ServerMessage()
    msg2.ParseFromString(data)

    assert msg2.conversation.voice_agent.listening == True
    assert msg2.conversation.voice_agent.last_command == "blur the laptop"
    assert msg2.conversation.voice_agent.full_screen_filter.filter_type == "dim"
    assert msg2.conversation.voice_agent.full_screen_filter.intensity == 0.5

    print("  ✓ VoiceAgentState protobuf works")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Voice Agent Pipeline — Smoke Tests")
    print("=" * 60)
    print()

    tests = [
        test_wake_word_gate,
        test_utterance_assembler,
        test_command_planner_fallback,
        test_effect_data,
        test_segment_data_with_effect,
        test_voice_agent_creation,
        test_protobuf_effect_fields,
        test_voice_agent_state_protobuf,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ All smoke tests passed!")
        print()
        print("Next steps:")
        print("  1. Start server: python -u -m server.main --device cuda --pipeline sam3")
        print("  2. Start client: python -m server.test_client_voice")
        print("  3. Say: 'Hey Vibe, blur the laptop, thank you'")
        return 0
    else:
        print("✗ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
