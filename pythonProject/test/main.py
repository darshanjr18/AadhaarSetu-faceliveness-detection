import cv2
from haar_cascade import load_face_cascade, detect_faces
from pca_module import load_pca_model, preprocess_face, apply_pca
from svm_module import load_svm_model, load_scaler, predict_liveness

# File paths for the models
PCA_MODEL_PATH = r'C:\Users\priya\LiveFaces\pythonProject\test\pca_model_compressed.pkl'
SVM_MODEL_PATH = r'C:\Users\priya\LiveFaces\pythonProject\test\svm_model.pkl'
SCALER_PATH = r'C:\Users\priya\LiveFaces\pythonProject\test\scaler.pkl'

# Load the models
pca = load_pca_model(PCA_MODEL_PATH)
svm = load_svm_model(SVM_MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# Load Haar Cascade
face_cascade = load_face_cascade()

def show_frame(frame):
    """Display the frame using OpenCV."""
    cv2.imshow('Face Liveness Detection', frame)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detect_faces(face_cascade, gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_img = gray[y:y + h, x:x + w]
        face_features = preprocess_face(face_img)
        face_features_scaled = scaler.transform([face_features])
        face_features_pca = apply_pca(pca, face_features_scaled)
        prediction = predict_liveness(svm, face_features_pca)

        if prediction == 1:
            label = ""
            color = (0, 255, 0)  # Green
        else:
            label = ""
            color = (0, 0, 255)  # Red

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the resulting frame
    show_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
