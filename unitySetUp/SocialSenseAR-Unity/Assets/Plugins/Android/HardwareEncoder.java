// No package - Unity finds it easier this way
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Hardware-accelerated encoder for Quest.
 * Uses MediaCodec for fast H.264 encoding or Bitmap for JPEG.
 */
public class HardwareEncoder {
    private static final String TAG = "HardwareEncoder";

    private MediaCodec encoder;
    private int width;
    private int height;
    private int quality = 92;  // Default high quality
    private boolean initialized = false;
    private ByteArrayOutputStream outputStream;
    private byte[] outputBuffer;

    // For async encoding
    private BlockingQueue<byte[]> inputQueue;
    private BlockingQueue<byte[]> outputQueue;
    private Thread encoderThread;
    private volatile boolean running = false;

    // Singleton instance
    private static HardwareEncoder instance;

    public static HardwareEncoder getInstance() {
        if (instance == null) {
            instance = new HardwareEncoder();
        }
        return instance;
    }

    /**
     * Initialize the hardware encoder.
     * @param w Width of frames
     * @param h Height of frames
     * @return true if successful
     */
    public boolean initialize(int w, int h) {
        Log.i(TAG, "initialize() called with " + w + "x" + h);
        this.width = w;
        this.height = h;

        try {
            // Find hardware JPEG encoder if available (optional - we use Bitmap.compress)
            MediaCodecList codecList = new MediaCodecList(MediaCodecList.REGULAR_CODECS);
            String jpegEncoderName = null;

            for (MediaCodecInfo info : codecList.getCodecInfos()) {
                if (!info.isEncoder()) continue;
                for (String type : info.getSupportedTypes()) {
                    if (type.equalsIgnoreCase("image/jpeg")) {
                        if (info.isHardwareAccelerated()) {
                            jpegEncoderName = info.getName();
                            Log.i(TAG, "Found hardware JPEG encoder: " + jpegEncoderName);
                            break;
                        }
                    }
                }
            }

            // Skip MediaCodec for now - Bitmap.compress is simpler and works well on Quest
            if (jpegEncoderName != null) {
                Log.i(TAG, "Hardware JPEG codec available but using Bitmap.compress for simplicity");
            } else {
                Log.i(TAG, "No hardware JPEG codec, using Bitmap.compress");
            }

            outputStream = new ByteArrayOutputStream(width * height);
            outputBuffer = new byte[width * height * 3];
            Log.i(TAG, "Allocated output buffers");

            // Setup async queues
            inputQueue = new ArrayBlockingQueue<>(2);
            outputQueue = new ArrayBlockingQueue<>(2);
            Log.i(TAG, "Created async queues");

            initialized = true;
            Log.i(TAG, "Initialization complete! Ready to encode " + width + "x" + height);
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize encoder: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Set JPEG quality (0-100).
     */
    public void setQuality(int q) {
        this.quality = Math.max(0, Math.min(100, q));
        Log.i(TAG, "Quality set to " + this.quality);
    }

    /**
     * Start the async encoding thread.
     */
    public void startAsync() {
        if (running) return;

        running = true;
        encoderThread = new Thread(() -> {
            Log.i(TAG, "Encoder thread started");
            while (running) {
                try {
                    byte[] pixels = inputQueue.poll(100, TimeUnit.MILLISECONDS);
                    if (pixels != null) {
                        byte[] encoded = encodeSync(pixels);
                        if (encoded != null) {
                            // Remove old output if queue is full
                            while (!outputQueue.offer(encoded)) {
                                outputQueue.poll();
                            }
                        }
                    }
                } catch (InterruptedException e) {
                    break;
                }
            }
            Log.i(TAG, "Encoder thread stopped");
        });
        encoderThread.start();
    }

    /**
     * Stop the async encoding thread.
     */
    public void stopAsync() {
        running = false;
        if (encoderThread != null) {
            try {
                encoderThread.join(1000);
            } catch (InterruptedException e) {
                // ignore
            }
        }
    }

    /**
     * Submit pixels for async encoding.
     * @param rgbPixels RGB24 pixel data
     */
    public void submitAsync(byte[] rgbPixels) {
        if (!running || inputQueue == null) return;

        // Non-blocking offer, drops frame if queue is full
        if (!inputQueue.offer(rgbPixels)) {
            // Queue full, drop oldest
            inputQueue.poll();
            inputQueue.offer(rgbPixels);
        }
    }

    /**
     * Get the latest encoded frame (non-blocking).
     * @return JPEG bytes or null if not ready
     */
    public byte[] getEncodedAsync() {
        if (outputQueue == null) return null;
        return outputQueue.poll();
    }

    // Reusable bitmap and int array to avoid allocation
    private Bitmap reusableBitmap;
    private int[] reusableIntPixels;

    /**
     * Encode RGB pixels to JPEG synchronously.
     * Uses hardware acceleration via Bitmap.compress when available.
     *
     * @param rgbPixels RGB24 pixel data (width * height * 3 bytes)
     * @return JPEG encoded bytes
     */
    public byte[] encodeSync(byte[] rgbPixels) {
        if (!initialized) {
            Log.e(TAG, "encodeSync called but not initialized!");
            return null;
        }

        if (rgbPixels == null) {
            Log.e(TAG, "encodeSync: rgbPixels is null!");
            return null;
        }

        int expectedSize = width * height * 3;
        if (rgbPixels.length != expectedSize) {
            Log.e(TAG, "encodeSync: size mismatch! Got " + rgbPixels.length + ", expected " + expectedSize + " (" + width + "x" + height + "x3)");
            return null;
        }

        try {
            // Reuse int array to avoid allocation
            if (reusableIntPixels == null || reusableIntPixels.length != width * height) {
                reusableIntPixels = new int[width * height];
            }

            // FAST RGB24 to ARGB8888 conversion
            int numPixels = width * height;
            for (int i = 0; i < numPixels; i++) {
                int idx = i * 3;
                reusableIntPixels[i] = 0xFF000000
                    | ((rgbPixels[idx] & 0xFF) << 16)
                    | ((rgbPixels[idx + 1] & 0xFF) << 8)
                    | (rgbPixels[idx + 2] & 0xFF);
            }

            // Reuse bitmap to avoid allocation
            if (reusableBitmap == null || reusableBitmap.getWidth() != width || reusableBitmap.getHeight() != height) {
                if (reusableBitmap != null) reusableBitmap.recycle();
                reusableBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            }
            reusableBitmap.setPixels(reusableIntPixels, 0, width, 0, 0, width, height);

            // Compress to JPEG (uses hardware when available on Quest)
            outputStream.reset();
            reusableBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);

            return outputStream.toByteArray();

        } catch (Exception e) {
            Log.e(TAG, "Encode error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * FAST: Encode RGBA pixels to JPEG (skips RGB conversion).
     * Use this when you have RGBA data from GPU readback.
     */
    public byte[] encodeSyncRGBA(byte[] rgbaPixels) {
        if (!initialized) return null;

        int expectedSize = width * height * 4;
        if (rgbaPixels == null || rgbaPixels.length != expectedSize) {
            Log.e(TAG, "encodeSyncRGBA: size mismatch!");
            return null;
        }

        try {
            // Reuse int array
            if (reusableIntPixels == null || reusableIntPixels.length != width * height) {
                reusableIntPixels = new int[width * height];
            }

            // FAST RGBA to ARGB conversion (just reorder bytes)
            int numPixels = width * height;
            for (int i = 0; i < numPixels; i++) {
                int idx = i * 4;
                // RGBA -> ARGB: swap A from end to beginning
                reusableIntPixels[i] = ((rgbaPixels[idx + 3] & 0xFF) << 24)  // A
                    | ((rgbaPixels[idx] & 0xFF) << 16)      // R
                    | ((rgbaPixels[idx + 1] & 0xFF) << 8)   // G
                    | (rgbaPixels[idx + 2] & 0xFF);         // B
            }

            // Reuse bitmap
            if (reusableBitmap == null || reusableBitmap.getWidth() != width || reusableBitmap.getHeight() != height) {
                if (reusableBitmap != null) reusableBitmap.recycle();
                reusableBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            }
            reusableBitmap.setPixels(reusableIntPixels, 0, width, 0, 0, width, height);

            // Compress to JPEG
            outputStream.reset();
            reusableBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);

            return outputStream.toByteArray();

        } catch (Exception e) {
            Log.e(TAG, "Encode RGBA error: " + e.getMessage());
            return null;
        }
    }

    /**
     * Fast JPEG encode using native buffer (avoids copy).
     * @param pixelBuffer Direct ByteBuffer with RGB24 data
     * @param quality JPEG quality 0-100
     * @return JPEG bytes
     */
    public byte[] encodeFromBuffer(ByteBuffer pixelBuffer, int quality) {
        if (!initialized) return null;

        try {
            // Read pixels from buffer
            pixelBuffer.rewind();
            int numPixels = width * height;
            int[] intPixels = new int[numPixels];

            for (int i = 0; i < numPixels; i++) {
                int r = pixelBuffer.get() & 0xFF;
                int g = pixelBuffer.get() & 0xFF;
                int b = pixelBuffer.get() & 0xFF;
                intPixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }

            Bitmap bitmap = Bitmap.createBitmap(intPixels, width, height, Bitmap.Config.ARGB_8888);

            outputStream.reset();
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);
            bitmap.recycle();

            return outputStream.toByteArray();

        } catch (Exception e) {
            Log.e(TAG, "Encode from buffer error: " + e.getMessage());
            return null;
        }
    }

    /**
     * Release encoder resources.
     */
    public void release() {
        stopAsync();

        if (encoder != null) {
            try {
                encoder.stop();
                encoder.release();
            } catch (Exception e) {
                // ignore
            }
            encoder = null;
        }

        initialized = false;
        instance = null;
        Log.i(TAG, "Encoder released");
    }

    /**
     * Check if hardware JPEG encoding is available.
     */
    public static boolean isHardwareJpegAvailable() {
        MediaCodecList codecList = new MediaCodecList(MediaCodecList.REGULAR_CODECS);
        for (MediaCodecInfo info : codecList.getCodecInfos()) {
            if (!info.isEncoder()) continue;
            for (String type : info.getSupportedTypes()) {
                if (type.equalsIgnoreCase("image/jpeg") && info.isHardwareAccelerated()) {
                    return true;
                }
            }
        }
        return false;
    }
}
