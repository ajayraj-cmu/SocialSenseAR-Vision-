import cv2
from deepface import DeepFace
import threading

# Configuration for FPS optimization
PROCESS_EVERY_N_FRAMES = 5
PROCESSING_WIDTH = 320

# Global variables for thread-safe communication
emotion_result = "Processing..."
emotion_lock = threading.Lock()
frame_count = 0

def process_emotion(frame):
    """Process emotion detection in a separate thread"""
    global emotion_result, emotion_lock
    
    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (PROCESSING_WIDTH, int(frame.shape[0] * PROCESSING_WIDTH / frame.shape[1])))
        
        # Analyze frame for emotions
        result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, silent=True)
        
        # Extract dominant emotion
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']
        
        with emotion_lock:
            emotion_result = emotion
    except Exception as e:
        with emotion_lock:
            emotion_result = "No face detected"

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting emotion detection. Press 'q' to quit.")

processing_thread = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process emotion detection every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        if processing_thread is None or not processing_thread.is_alive():
            processing_thread = threading.Thread(target=process_emotion, args=(frame.copy(),))
            processing_thread.daemon = True
            processing_thread.start()
    
    # Get current emotion result
    with emotion_lock:
        current_emotion = emotion_result
    
    # Map happy and neutral to calm for display
    display_emotion = "calm" if current_emotion.lower() in ["happy", "neutral"] else current_emotion
    
    # Display emotion on frame
    color = (0, 255, 0) if display_emotion != "Processing..." and display_emotion != "No face detected" else (0, 0, 255)
    cv2.putText(frame, f"Emotion: {display_emotion}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()