import cv2

def load_face_cascade():
    """Load Haar Cascade for face detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces(face_cascade, gray_frame):
    """Detect faces in a grayscale frame."""
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces
