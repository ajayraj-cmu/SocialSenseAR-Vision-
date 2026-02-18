import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.util.Log;
import android.view.Surface;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Hardware H.264 encoder for Quest - encodes frames in ~2-3ms.
 * Uses MediaCodec hardware encoder for real-time 4K60 encoding.
 */
public class H264HardwareEncoder {
    private static final String TAG = "H264HWEncoder";
    private static final String MIME_TYPE = "video/avc";  // H.264

    private MediaCodec encoder;
    private int width, height;
    private boolean initialized = false;
    private boolean encoding = false;

    // Output queue for encoded frames
    private ArrayBlockingQueue<byte[]> outputQueue = new ArrayBlockingQueue<>(4);

    // Frame header for Python: [4-byte size][1-byte type=0x10][data]
    private static final byte FRAME_TYPE_H264 = 0x10;

    // Singleton
    private static H264HardwareEncoder instance;

    public static H264HardwareEncoder getInstance() {
        if (instance == null) {
            instance = new H264HardwareEncoder();
        }
        return instance;
    }

    /**
     * Initialize the H.264 hardware encoder.
     */
    public boolean initialize(int w, int h, int fps, int bitrate) {
        width = w;
        height = h;

        try {
            // Find hardware H.264 encoder
            String encoderName = null;
            MediaCodecList codecList = new MediaCodecList(MediaCodecList.REGULAR_CODECS);

            for (MediaCodecInfo info : codecList.getCodecInfos()) {
                if (!info.isEncoder()) continue;
                for (String type : info.getSupportedTypes()) {
                    if (type.equalsIgnoreCase(MIME_TYPE)) {
                        if (info.isHardwareAccelerated()) {
                            encoderName = info.getName();
                            Log.i(TAG, "Found HW H.264 encoder: " + encoderName);
                            break;
                        }
                    }
                }
                if (encoderName != null) break;
            }

            if (encoderName == null) {
                Log.e(TAG, "No hardware H.264 encoder found!");
                return false;
            }

            // Create encoder
            encoder = MediaCodec.createByCodecName(encoderName);

            // Configure format
            MediaFormat format = MediaFormat.createVideoFormat(MIME_TYPE, width, height);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            format.setInteger(MediaFormat.KEY_BIT_RATE, bitrate);
            format.setInteger(MediaFormat.KEY_FRAME_RATE, fps);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);  // Keyframe every 1 second

            // Low latency settings
            format.setInteger(MediaFormat.KEY_LATENCY, 0);
            format.setInteger(MediaFormat.KEY_PRIORITY, 0);  // Real-time priority

            // Try to set low latency mode (API 30+)
            try {
                format.setInteger("vendor.qti-ext-enc-low-latency.enable", 1);
            } catch (Exception e) {
                // Ignore if not supported
            }

            encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            encoder.start();

            initialized = true;
            encoding = true;

            // Start output thread
            new Thread(this::outputLoop, "H264-Output").start();

            Log.i(TAG, "H.264 encoder initialized: " + width + "x" + height + " @ " + fps + "fps, " + (bitrate/1000000) + "Mbps");
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize H.264 encoder: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Encode a frame (RGBA pixels).
     * Returns immediately, encoded data available via getEncodedFrame().
     */
    public void encodeFrame(byte[] rgbaPixels) {
        if (!initialized || encoder == null) return;

        try {
            // Get input buffer
            int inputIndex = encoder.dequeueInputBuffer(10000);  // 10ms timeout
            if (inputIndex < 0) {
                Log.w(TAG, "No input buffer available");
                return;
            }

            ByteBuffer inputBuffer = encoder.getInputBuffer(inputIndex);
            if (inputBuffer == null) return;

            // Convert RGBA to NV12 (YUV420) for encoder
            byte[] nv12 = rgbaToNV12(rgbaPixels, width, height);

            inputBuffer.clear();
            inputBuffer.put(nv12);

            // Queue input buffer
            long pts = System.nanoTime() / 1000;  // Presentation time in microseconds
            encoder.queueInputBuffer(inputIndex, 0, nv12.length, pts, 0);

        } catch (Exception e) {
            Log.e(TAG, "Encode error: " + e.getMessage());
        }
    }

    /**
     * Output loop - runs in background thread.
     */
    private void outputLoop() {
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        ByteArrayOutputStream frameBuffer = new ByteArrayOutputStream();

        while (encoding) {
            try {
                int outputIndex = encoder.dequeueOutputBuffer(info, 10000);

                if (outputIndex >= 0) {
                    ByteBuffer outputBuffer = encoder.getOutputBuffer(outputIndex);
                    if (outputBuffer != null && info.size > 0) {
                        // Read encoded data
                        byte[] encodedData = new byte[info.size];
                        outputBuffer.get(encodedData);

                        // Add frame type header
                        frameBuffer.reset();
                        frameBuffer.write(FRAME_TYPE_H264);
                        frameBuffer.write(encodedData);

                        // Queue for retrieval
                        byte[] frame = frameBuffer.toByteArray();
                        if (!outputQueue.offer(frame)) {
                            outputQueue.poll();  // Drop oldest
                            outputQueue.offer(frame);
                        }
                    }
                    encoder.releaseOutputBuffer(outputIndex, false);
                }
            } catch (Exception e) {
                if (encoding) {
                    Log.e(TAG, "Output error: " + e.getMessage());
                }
            }
        }
    }

    /**
     * Get encoded frame (non-blocking).
     */
    public byte[] getEncodedFrame() {
        return outputQueue.poll();
    }

    /**
     * Convert RGBA to NV12 (YUV420 semi-planar).
     * This is what the hardware encoder expects.
     */
    private byte[] rgbaToNV12(byte[] rgba, int w, int h) {
        int frameSize = w * h;
        int chromaSize = frameSize / 2;
        byte[] nv12 = new byte[frameSize + chromaSize];

        int yIndex = 0;
        int uvIndex = frameSize;

        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                int rgbaIndex = (j * w + i) * 4;
                int r = rgba[rgbaIndex] & 0xFF;
                int g = rgba[rgbaIndex + 1] & 0xFF;
                int b = rgba[rgbaIndex + 2] & 0xFF;

                // RGB to YUV conversion
                int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                nv12[yIndex++] = (byte) Math.max(0, Math.min(255, y));

                // Subsample UV (every 2x2 block)
                if (j % 2 == 0 && i % 2 == 0) {
                    int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                    nv12[uvIndex++] = (byte) Math.max(0, Math.min(255, u));
                    nv12[uvIndex++] = (byte) Math.max(0, Math.min(255, v));
                }
            }
        }

        return nv12;
    }

    /**
     * Release encoder resources.
     */
    public void release() {
        encoding = false;

        if (encoder != null) {
            try {
                encoder.stop();
                encoder.release();
            } catch (Exception e) {
                // Ignore
            }
            encoder = null;
        }

        initialized = false;
        instance = null;
        Log.i(TAG, "H.264 encoder released");
    }

    /**
     * Check if hardware H.264 encoding is available.
     */
    public static boolean isAvailable() {
        MediaCodecList codecList = new MediaCodecList(MediaCodecList.REGULAR_CODECS);
        for (MediaCodecInfo info : codecList.getCodecInfos()) {
            if (!info.isEncoder()) continue;
            for (String type : info.getSupportedTypes()) {
                if (type.equalsIgnoreCase(MIME_TYPE) && info.isHardwareAccelerated()) {
                    return true;
                }
            }
        }
        return false;
    }
}
