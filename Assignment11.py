import cv2
import pytesseract
import requests
import numpy as np
from fer import FER  # Lightweight emotion recognition

# -----------------------------
# Part 1: Medicine Label Reader
# -----------------------------
def read_medicine_label_from_url(url):
    print("[Medicine Label Reader] Downloading image...")
    resp = requests.get(url)
    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to improve OCR accuracy
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # OCR config
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    print("\n[Medicine Label Text Extracted]:\n")
    print(text)
    return text

# -----------------------------
# Part 2: Emotion-aware Assistant
# -----------------------------
def emotion_detection_from_webcam():
    print("\n[Emotion-aware Assistant] Starting webcam for emotion detection...")
    cap = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)  # Use MTCNN for better face detection
    
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions
        results = detector.detect_emotions(frame)
        
        # Draw bounding boxes and labels
        for face in results:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Emotion-aware Cognitive Assistant", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("[Emotion-aware Assistant] Stopped webcam.")

# -----------------------------
# Run both parts
# -----------------------------

# Medicine label image URL example (you can replace with any medicine label image URL)
medicine_label_url = 'https://upload.wikimedia.org/wikipedia/commons/8/82/Medicine_Bottle_Label.jpg'

# Read medicine label text
read_medicine_label_from_url(medicine_label_url)

# Start emotion detection (webcam)
emotion_detection_from_webcam()
