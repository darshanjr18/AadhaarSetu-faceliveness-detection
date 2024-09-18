import joblib
import cv2

# Define image size (should match the one used in training)
IMAGE_SIZE = (32, 32)

def load_pca_model(pca_model_path):
    """Load the PCA model from file."""
    pca = joblib.load(pca_model_path)
    return pca

def preprocess_face(face_img):
    """Resize and flatten the face image for PCA."""
    img_resized = cv2.resize(face_img, IMAGE_SIZE)
    img_flattened = img_resized.flatten()
    return img_flattened

def apply_pca(pca_model, scaled_features):
    """Apply PCA transformation to the scaled face features."""
    pca_features = pca_model.transform(scaled_features)
    return pca_features
