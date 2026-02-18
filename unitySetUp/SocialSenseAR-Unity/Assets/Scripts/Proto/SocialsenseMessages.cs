// Hand-written protobuf-compatible C# classes for socialsense.proto.
// Uses Google.Protobuf runtime for serialization/deserialization.
// Field numbers and wire types match the .proto schema exactly.
//
// Protobuf tag formula: (field_number << 3) | wire_type
//   wire_type 0 = varint (uint32, uint64, bool)
//   wire_type 1 = 64-bit (double)
//   wire_type 2 = length-delimited (string, bytes, message)
//   wire_type 5 = 32-bit (float)

using Google.Protobuf;
using Google.Protobuf.Collections;
using Google.Protobuf.Reflection;
using System;
using System.Collections.Generic;

namespace Socialsense
{
    // ================================================================
    // CLIENT -> SERVER (send only — WriteTo + CalculateSize needed)
    // ================================================================

    public sealed class FramePayload : IMessage<FramePayload>
    {
        public ByteString JpegData { get; set; } = ByteString.Empty;
        public uint Width { get; set; }
        public uint Height { get; set; }
        public uint Quality { get; set; }

        public MessageDescriptor Descriptor => null;
        public FramePayload() { }
        public FramePayload(FramePayload other) { MergeFrom(other); }
        public FramePayload Clone() => new FramePayload(this);

        public int CalculateSize()
        {
            int size = 0;
            if (JpegData.Length > 0) size += 1 + CodedOutputStream.ComputeBytesSize(JpegData);
            if (Width != 0) size += 1 + CodedOutputStream.ComputeUInt32Size(Width);
            if (Height != 0) size += 1 + CodedOutputStream.ComputeUInt32Size(Height);
            if (Quality != 0) size += 1 + CodedOutputStream.ComputeUInt32Size(Quality);
            return size;
        }

        public void WriteTo(CodedOutputStream output)
        {
            if (JpegData.Length > 0) { output.WriteRawTag(10); output.WriteBytes(JpegData); }     // field 1, wire 2
            if (Width != 0) { output.WriteRawTag(16); output.WriteUInt32(Width); }                 // field 2, wire 0
            if (Height != 0) { output.WriteRawTag(24); output.WriteUInt32(Height); }               // field 3, wire 0
            if (Quality != 0) { output.WriteRawTag(32); output.WriteUInt32(Quality); }             // field 4, wire 0
        }

        public void MergeFrom(FramePayload other)
        {
            if (other == null) return;
            if (other.JpegData.Length > 0) JpegData = other.JpegData;
            if (other.Width != 0) Width = other.Width;
            if (other.Height != 0) Height = other.Height;
            if (other.Quality != 0) Quality = other.Quality;
        }

        public void MergeFrom(CodedInputStream input) { /* send-only */ }
        public bool Equals(FramePayload other) => other != null;
        public override bool Equals(object obj) => Equals(obj as FramePayload);
        public override int GetHashCode() => (int)Width;
    }

    public sealed class AudioPayload : IMessage<AudioPayload>
    {
        public ByteString Pcm16Data { get; set; } = ByteString.Empty;
        public uint SampleRate { get; set; }
        public uint NumSamples { get; set; }

        public MessageDescriptor Descriptor => null;
        public AudioPayload() { }
        public AudioPayload(AudioPayload other) { MergeFrom(other); }
        public AudioPayload Clone() => new AudioPayload(this);

        public int CalculateSize()
        {
            int size = 0;
            if (Pcm16Data.Length > 0) size += 1 + CodedOutputStream.ComputeBytesSize(Pcm16Data);
            if (SampleRate != 0) size += 1 + CodedOutputStream.ComputeUInt32Size(SampleRate);
            if (NumSamples != 0) size += 1 + CodedOutputStream.ComputeUInt32Size(NumSamples);
            return size;
        }

        public void WriteTo(CodedOutputStream output)
        {
            if (Pcm16Data.Length > 0) { output.WriteRawTag(10); output.WriteBytes(Pcm16Data); }   // field 1, wire 2
            if (SampleRate != 0) { output.WriteRawTag(16); output.WriteUInt32(SampleRate); }       // field 2, wire 0
            if (NumSamples != 0) { output.WriteRawTag(24); output.WriteUInt32(NumSamples); }       // field 3, wire 0
        }

        public void MergeFrom(AudioPayload other)
        {
            if (other == null) return;
            if (other.Pcm16Data.Length > 0) Pcm16Data = other.Pcm16Data;
            if (other.SampleRate != 0) SampleRate = other.SampleRate;
            if (other.NumSamples != 0) NumSamples = other.NumSamples;
        }

        public void MergeFrom(CodedInputStream input) { /* send-only */ }
        public bool Equals(AudioPayload other) => other != null;
        public override bool Equals(object obj) => Equals(obj as AudioPayload);
        public override int GetHashCode() => (int)SampleRate;
    }

    public sealed class ControlPayload : IMessage<ControlPayload>
    {
        public string Command { get; set; } = "";

        public MessageDescriptor Descriptor => null;
        public ControlPayload() { }
        public ControlPayload(ControlPayload other) { MergeFrom(other); }
        public ControlPayload Clone() => new ControlPayload(this);

        public int CalculateSize()
        {
            int size = 0;
            if (Command.Length > 0) size += 1 + CodedOutputStream.ComputeStringSize(Command);
            return size;
        }

        public void WriteTo(CodedOutputStream output)
        {
            if (Command.Length > 0) { output.WriteRawTag(10); output.WriteString(Command); }       // field 1, wire 2
        }

        public void MergeFrom(ControlPayload other)
        {
            if (other == null) return;
            if (other.Command.Length > 0) Command = other.Command;
        }

        public void MergeFrom(CodedInputStream input) { /* send-only */ }
        public bool Equals(ControlPayload other) => other != null;
        public override bool Equals(object obj) => Equals(obj as ControlPayload);
        public override int GetHashCode() => Command.GetHashCode();
    }

    public sealed class ClientMessage : IMessage<ClientMessage>
    {
        public ulong FrameId { get; set; }
        public double TimestampMs { get; set; }

        // oneof payload { frame=3, audio=4, control=5 }
        public FramePayload Frame { get; set; }
        public AudioPayload Audio { get; set; }
        public ControlPayload Control { get; set; }

        public MessageDescriptor Descriptor => null;
        public ClientMessage() { }
        public ClientMessage(ClientMessage other) { MergeFrom(other); }
        public ClientMessage Clone() => new ClientMessage(this);

        public int CalculateSize()
        {
            int size = 0;
            if (FrameId != 0) size += 1 + CodedOutputStream.ComputeUInt64Size(FrameId);
            if (TimestampMs != 0) size += 1 + 8; // tag + fixed64
            if (Frame != null) size += 1 + CodedOutputStream.ComputeMessageSize(Frame);
            if (Audio != null) size += 1 + CodedOutputStream.ComputeMessageSize(Audio);
            if (Control != null) size += 1 + CodedOutputStream.ComputeMessageSize(Control);
            return size;
        }

        public void WriteTo(CodedOutputStream output)
        {
            if (FrameId != 0) { output.WriteRawTag(8); output.WriteUInt64(FrameId); }             // field 1, wire 0
            if (TimestampMs != 0) { output.WriteRawTag(17); output.WriteDouble(TimestampMs); }     // field 2, wire 1
            if (Frame != null) { output.WriteRawTag(26); output.WriteMessage(Frame); }             // field 3, wire 2
            if (Audio != null) { output.WriteRawTag(34); output.WriteMessage(Audio); }             // field 4, wire 2
            if (Control != null) { output.WriteRawTag(42); output.WriteMessage(Control); }         // field 5, wire 2
        }

        public void MergeFrom(ClientMessage other)
        {
            if (other == null) return;
            if (other.FrameId != 0) FrameId = other.FrameId;
            if (other.TimestampMs != 0) TimestampMs = other.TimestampMs;
            if (other.Frame != null) Frame = other.Frame.Clone();
            if (other.Audio != null) Audio = other.Audio.Clone();
            if (other.Control != null) Control = other.Control.Clone();
        }

        public void MergeFrom(CodedInputStream input) { /* send-only */ }
        public bool Equals(ClientMessage other) => other != null && FrameId == other.FrameId;
        public override bool Equals(object obj) => Equals(obj as ClientMessage);
        public override int GetHashCode() => FrameId.GetHashCode();
    }

    // ================================================================
    // SERVER -> CLIENT (receive only — MergeFrom needed)
    // ================================================================

    public sealed class BoundingBox : IMessage<BoundingBox>
    {
        public float XMin { get; set; }
        public float YMin { get; set; }
        public float XMax { get; set; }
        public float YMax { get; set; }

        public MessageDescriptor Descriptor => null;
        public BoundingBox() { }
        public BoundingBox(BoundingBox other) { MergeFrom(other); }
        public BoundingBox Clone() => new BoundingBox(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(BoundingBox other)
        {
            if (other == null) return;
            XMin = other.XMin; YMin = other.YMin;
            XMax = other.XMax; YMax = other.YMax;
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 13: XMin = input.ReadFloat(); break;  // field 1, wire 5
                    case 21: YMin = input.ReadFloat(); break;  // field 2, wire 5
                    case 29: XMax = input.ReadFloat(); break;  // field 3, wire 5
                    case 37: YMax = input.ReadFloat(); break;  // field 4, wire 5
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(BoundingBox other) => other != null;
        public override bool Equals(object obj) => Equals(obj as BoundingBox);
        public override int GetHashCode() => 0;
    }

    public sealed class EmotionResult : IMessage<EmotionResult>
    {
        public string PrimaryEmotion { get; set; } = "";
        public string DisplayLabel { get; set; } = "";
        public string Emoji { get; set; } = "";
        public float Confidence { get; set; }

        public MessageDescriptor Descriptor => null;
        public EmotionResult() { }
        public EmotionResult(EmotionResult other) { MergeFrom(other); }
        public EmotionResult Clone() => new EmotionResult(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(EmotionResult other)
        {
            if (other == null) return;
            PrimaryEmotion = other.PrimaryEmotion;
            DisplayLabel = other.DisplayLabel;
            Emoji = other.Emoji;
            Confidence = other.Confidence;
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 10: PrimaryEmotion = input.ReadString(); break;  // field 1, wire 2
                    case 18: DisplayLabel = input.ReadString(); break;    // field 2, wire 2
                    case 26: Emoji = input.ReadString(); break;           // field 3, wire 2
                    case 37: Confidence = input.ReadFloat(); break;       // field 4, wire 5
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(EmotionResult other) => other != null;
        public override bool Equals(object obj) => Equals(obj as EmotionResult);
        public override int GetHashCode() => 0;
    }

    public sealed class EffectMetadata : IMessage<EffectMetadata>
    {
        public string EffectType { get; set; } = "";    // "blur", "dim", "pixelate", "highlight", "outline", "none"
        public float Intensity { get; set; }              // 0.0-1.0
        public string ColorHex { get; set; } = "";        // hex color for outline/highlight (e.g., "#FF0000")
        public Dictionary<string, string> Params { get; set; } = new Dictionary<string, string>();

        // Convenience: true when effect applies to everything OUTSIDE the mask
        public bool Invert => Params.TryGetValue("invert", out string v) && v == "true";

        public MessageDescriptor Descriptor => null;
        public EffectMetadata() { }
        public EffectMetadata(EffectMetadata other) { MergeFrom(other); }
        public EffectMetadata Clone() => new EffectMetadata(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(EffectMetadata other)
        {
            if (other == null) return;
            if (other.EffectType.Length > 0) EffectType = other.EffectType;
            if (other.Intensity != 0) Intensity = other.Intensity;
            if (other.ColorHex.Length > 0) ColorHex = other.ColorHex;
            foreach (var kv in other.Params) Params[kv.Key] = kv.Value;
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 10: EffectType = input.ReadString(); break;    // field 1, wire 2
                    case 21: Intensity = input.ReadFloat(); break;      // field 2, wire 5
                    case 26: ColorHex = input.ReadString(); break;      // field 3, wire 2
                    case 34:                                             // field 4, wire 2 (map entry)
                        // Protobuf map<string,string> = repeated message { string key=1; string value=2; }
                        ReadMapEntry(input);
                        break;
                    default: input.SkipLastField(); break;
                }
            }
        }

        private void ReadMapEntry(CodedInputStream input)
        {
            // ReadBytes() handles the length-prefix and returns the content
            ByteString entryData = input.ReadBytes();

            string key = "", value = "";
            var entryInput = new CodedInputStream(entryData.ToByteArray());
            uint entryTag;
            while ((entryTag = entryInput.ReadTag()) != 0)
            {
                switch (entryTag)
                {
                    case 10: key = entryInput.ReadString(); break;     // entry field 1
                    case 18: value = entryInput.ReadString(); break;   // entry field 2
                    default: entryInput.SkipLastField(); break;
                }
            }
            if (key.Length > 0) Params[key] = value;
        }

        public bool Equals(EffectMetadata other) => other != null;
        public override bool Equals(object obj) => Equals(obj as EffectMetadata);
        public override int GetHashCode() => EffectType.GetHashCode();
    }

    public sealed class FullScreenFilter : IMessage<FullScreenFilter>
    {
        public string FilterType { get; set; } = "";    // "dim", "warm", "cool", "night", "grayscale", "color", "none"
        public float Intensity { get; set; }              // 0.0-1.0
        public string ColorHex { get; set; } = "";        // hex color if filter_type is "color" (e.g., "#0000FF")

        public MessageDescriptor Descriptor => null;
        public FullScreenFilter() { }
        public FullScreenFilter(FullScreenFilter other) { MergeFrom(other); }
        public FullScreenFilter Clone() => new FullScreenFilter(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(FullScreenFilter other)
        {
            if (other == null) return;
            if (other.FilterType.Length > 0) FilterType = other.FilterType;
            if (other.Intensity != 0) Intensity = other.Intensity;
            if (other.ColorHex.Length > 0) ColorHex = other.ColorHex;
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 10: FilterType = input.ReadString(); break;    // field 1, wire 2
                    case 21: Intensity = input.ReadFloat(); break;      // field 2, wire 5
                    case 26: ColorHex = input.ReadString(); break;      // field 3, wire 2
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(FullScreenFilter other) => other != null;
        public override bool Equals(object obj) => Equals(obj as FullScreenFilter);
        public override int GetHashCode() => FilterType.GetHashCode();
    }

    public sealed class VoiceAgentState : IMessage<VoiceAgentState>
    {
        public bool Listening { get; set; }
        public bool Recording { get; set; }
        public string PartialTranscript { get; set; } = "";
        public string LastCommand { get; set; } = "";
        public string LastResponse { get; set; } = "";
        public double LastCommandTime { get; set; }
        public FullScreenFilter FullScreenFilter { get; set; }

        public MessageDescriptor Descriptor => null;
        public VoiceAgentState() { }
        public VoiceAgentState(VoiceAgentState other) { MergeFrom(other); }
        public VoiceAgentState Clone() => new VoiceAgentState(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(VoiceAgentState other)
        {
            if (other == null) return;
            Listening = other.Listening;
            Recording = other.Recording;
            if (other.PartialTranscript.Length > 0) PartialTranscript = other.PartialTranscript;
            if (other.LastCommand.Length > 0) LastCommand = other.LastCommand;
            if (other.LastResponse.Length > 0) LastResponse = other.LastResponse;
            if (other.LastCommandTime != 0) LastCommandTime = other.LastCommandTime;
            if (other.FullScreenFilter != null) FullScreenFilter = other.FullScreenFilter.Clone();
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 8: Listening = input.ReadBool(); break;                // field 1, wire 0
                    case 16: Recording = input.ReadBool(); break;               // field 2, wire 0
                    case 26: PartialTranscript = input.ReadString(); break;     // field 3, wire 2
                    case 34: LastCommand = input.ReadString(); break;           // field 4, wire 2
                    case 42: LastResponse = input.ReadString(); break;          // field 5, wire 2
                    case 49: LastCommandTime = input.ReadDouble(); break;       // field 6, wire 1
                    case 82:                                                     // field 10, wire 2
                        if (FullScreenFilter == null) FullScreenFilter = new FullScreenFilter();
                        input.ReadMessage(FullScreenFilter);
                        break;
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(VoiceAgentState other) => other != null;
        public override bool Equals(object obj) => Equals(obj as VoiceAgentState);
        public override int GetHashCode() => 0;
    }

    public sealed class SceneSegment : IMessage<SceneSegment>
    {
        public string Label { get; set; } = "";
        public string AssetClass { get; set; } = "";
        public float Confidence { get; set; }
        public BoundingBox Bbox { get; set; }
        public ByteString RleMask { get; set; } = ByteString.Empty;
        public uint MaskWidth { get; set; }
        public uint MaskHeight { get; set; }
        public float CenterX { get; set; }
        public float CenterY { get; set; }
        public string TrackId { get; set; } = "";
        public EmotionResult Emotion { get; set; }
        public EffectMetadata Effect { get; set; }

        public MessageDescriptor Descriptor => null;
        public SceneSegment() { }
        public SceneSegment(SceneSegment other) { MergeFrom(other); }
        public SceneSegment Clone() => new SceneSegment(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(SceneSegment other)
        {
            if (other == null) return;
            Label = other.Label; AssetClass = other.AssetClass;
            Confidence = other.Confidence;
            if (other.Bbox != null) Bbox = other.Bbox.Clone();
            RleMask = other.RleMask;
            MaskWidth = other.MaskWidth; MaskHeight = other.MaskHeight;
            CenterX = other.CenterX; CenterY = other.CenterY;
            TrackId = other.TrackId;
            if (other.Emotion != null) Emotion = other.Emotion.Clone();
            if (other.Effect != null) Effect = other.Effect.Clone();
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 10: Label = input.ReadString(); break;            // field 1, wire 2
                    case 18: AssetClass = input.ReadString(); break;       // field 2, wire 2
                    case 29: Confidence = input.ReadFloat(); break;        // field 3, wire 5
                    case 34:                                                // field 4, wire 2
                        if (Bbox == null) Bbox = new BoundingBox();
                        input.ReadMessage(Bbox);
                        break;
                    case 42: RleMask = input.ReadBytes(); break;           // field 5, wire 2
                    case 48: MaskWidth = input.ReadUInt32(); break;        // field 6, wire 0
                    case 56: MaskHeight = input.ReadUInt32(); break;       // field 7, wire 0
                    case 69: CenterX = input.ReadFloat(); break;           // field 8, wire 5
                    case 77: CenterY = input.ReadFloat(); break;           // field 9, wire 5
                    case 82: TrackId = input.ReadString(); break;          // field 10, wire 2
                    case 122:                                               // field 15, wire 2
                        if (Emotion == null) Emotion = new EmotionResult();
                        input.ReadMessage(Emotion);
                        break;
                    case 130:                                               // field 16, wire 2
                        if (Effect == null) Effect = new EffectMetadata();
                        input.ReadMessage(Effect);
                        break;
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(SceneSegment other) => other != null;
        public override bool Equals(object obj) => Equals(obj as SceneSegment);
        public override int GetHashCode() => 0;
    }

    public sealed class PerformanceMetrics : IMessage<PerformanceMetrics>
    {
        public float FastsamMs { get; set; }
        public float LabelingMs { get; set; }
        public float EmotionMs { get; set; }
        public float AudioMs { get; set; }
        public float TotalMs { get; set; }
        public uint SegmentCount { get; set; }

        public MessageDescriptor Descriptor => null;
        public PerformanceMetrics() { }
        public PerformanceMetrics(PerformanceMetrics other) { MergeFrom(other); }
        public PerformanceMetrics Clone() => new PerformanceMetrics(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }
        public void MergeFrom(PerformanceMetrics other) { }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 13: FastsamMs = input.ReadFloat(); break;       // field 1, wire 5
                    case 21: LabelingMs = input.ReadFloat(); break;      // field 2, wire 5
                    case 29: EmotionMs = input.ReadFloat(); break;       // field 3, wire 5
                    case 37: AudioMs = input.ReadFloat(); break;         // field 4, wire 5
                    case 45: TotalMs = input.ReadFloat(); break;         // field 5, wire 5
                    case 48: SegmentCount = input.ReadUInt32(); break;   // field 6, wire 0
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(PerformanceMetrics other) => other != null;
        public override bool Equals(object obj) => Equals(obj as PerformanceMetrics);
        public override int GetHashCode() => 0;
    }

    public sealed class SocialCue : IMessage<SocialCue>
    {
        public string CueType { get; set; } = "";
        public string Icon { get; set; } = "";
        public string Message { get; set; } = "";
        public float Confidence { get; set; }
        public double Timestamp { get; set; }

        public MessageDescriptor Descriptor => null;
        public SocialCue() { }
        public SocialCue(SocialCue other) { MergeFrom(other); }
        public SocialCue Clone() => new SocialCue(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }
        public void MergeFrom(SocialCue other) { }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 10: CueType = input.ReadString(); break;     // field 1, wire 2
                    case 18: Icon = input.ReadString(); break;         // field 2, wire 2
                    case 26: Message = input.ReadString(); break;      // field 3, wire 2
                    case 37: Confidence = input.ReadFloat(); break;    // field 4, wire 5
                    case 41: Timestamp = input.ReadDouble(); break;    // field 5, wire 1
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(SocialCue other) => other != null;
        public override bool Equals(object obj) => Equals(obj as SocialCue);
        public override int GetHashCode() => 0;
    }

    public sealed class ConversationState : IMessage<ConversationState>
    {
        public string Summary { get; set; } = "";
        public string Vibe { get; set; } = "";
        public string RecentUtterance { get; set; } = "";
        public string Question { get; set; } = "";
        public bool IsOtherSpeaking { get; set; }
        public bool IsUserSpeaking { get; set; }
        public SocialCue SocialCueData { get; set; }
        public VoiceAgentState VoiceAgent { get; set; }

        public MessageDescriptor Descriptor => null;
        public ConversationState() { }
        public ConversationState(ConversationState other) { MergeFrom(other); }
        public ConversationState Clone() => new ConversationState(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }
        public void MergeFrom(ConversationState other) { }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 10: Summary = input.ReadString(); break;          // field 1, wire 2
                    case 18: Vibe = input.ReadString(); break;             // field 2, wire 2
                    case 26: RecentUtterance = input.ReadString(); break;  // field 3, wire 2
                    case 34: Question = input.ReadString(); break;         // field 4, wire 2
                    case 40: IsOtherSpeaking = input.ReadBool(); break;    // field 5, wire 0
                    case 48: IsUserSpeaking = input.ReadBool(); break;     // field 6, wire 0
                    case 82:                                                // field 10, wire 2
                        if (SocialCueData == null) SocialCueData = new SocialCue();
                        input.ReadMessage(SocialCueData);
                        break;
                    case 162:                                               // field 20, wire 2
                        if (VoiceAgent == null) VoiceAgent = new VoiceAgentState();
                        input.ReadMessage(VoiceAgent);
                        break;
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(ConversationState other) => other != null;
        public override bool Equals(object obj) => Equals(obj as ConversationState);
        public override int GetHashCode() => 0;
    }

    public sealed class ServerMessage : IMessage<ServerMessage>
    {
        public ulong FrameId { get; set; }
        public double TimestampMs { get; set; }
        public double ProcessingTimeMs { get; set; }
        public ulong MasksFrameId { get; set; }    // frame_id that masks actually correspond to
        public bool MasksUpdated { get; set; }       // true when SAM produced new masks
        public bool EffectsCleared { get; set; }     // true when all effects were explicitly cleared (distinguishes from "target not detected")
        public RepeatedField<SceneSegment> Segments { get; } = new RepeatedField<SceneSegment>();
        public ConversationState Conversation { get; set; }
        public PerformanceMetrics Metrics { get; set; }

        public MessageDescriptor Descriptor => null;
        public ServerMessage() { }
        public ServerMessage(ServerMessage other) { MergeFrom(other); }
        public ServerMessage Clone() => new ServerMessage(this);
        public int CalculateSize() => 0;
        public void WriteTo(CodedOutputStream output) { }

        public void MergeFrom(ServerMessage other)
        {
            if (other == null) return;
            FrameId = other.FrameId;
            TimestampMs = other.TimestampMs;
            ProcessingTimeMs = other.ProcessingTimeMs;
            MasksFrameId = other.MasksFrameId;
            MasksUpdated = other.MasksUpdated;
            EffectsCleared = other.EffectsCleared;
            Segments.Add(other.Segments);
        }

        public void MergeFrom(CodedInputStream input)
        {
            uint tag;
            while ((tag = input.ReadTag()) != 0)
            {
                switch (tag)
                {
                    case 8: FrameId = input.ReadUInt64(); break;            // field 1, wire 0
                    case 17: TimestampMs = input.ReadDouble(); break;       // field 2, wire 1
                    case 25: ProcessingTimeMs = input.ReadDouble(); break;   // field 3, wire 1
                    case 40: MasksFrameId = input.ReadUInt64(); break;      // field 5, wire 0
                    case 48: MasksUpdated = input.ReadBool(); break;        // field 6, wire 0
                    case 56: EffectsCleared = input.ReadBool(); break;      // field 7, wire 0
                    case 82:                                                 // field 10, wire 2
                        var seg = new SceneSegment();
                        input.ReadMessage(seg);
                        Segments.Add(seg);
                        break;
                    case 162:                                                // field 20, wire 2
                        if (Conversation == null) Conversation = new ConversationState();
                        input.ReadMessage(Conversation);
                        break;
                    case 242:                                                // field 30, wire 2
                        if (Metrics == null) Metrics = new PerformanceMetrics();
                        input.ReadMessage(Metrics);
                        break;
                    default: input.SkipLastField(); break;
                }
            }
        }

        public bool Equals(ServerMessage other) => other != null && FrameId == other.FrameId;
        public override bool Equals(object obj) => Equals(obj as ServerMessage);
        public override int GetHashCode() => FrameId.GetHashCode();
        public override string ToString() => $"ServerMessage(frame={FrameId}, segs={Segments.Count})";
    }
}
