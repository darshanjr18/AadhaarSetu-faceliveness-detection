import joblib

def load_svm_model(svm_model_path):
    """Load the trained SVM model from file."""
    svm = joblib.load(svm_model_path)
    return svm

def load_scaler(scaler_path):
    """Load the scaler model from file."""
    scaler = joblib.load(scaler_path)
    return scaler

def predict_liveness(svm_model, pca_features):
    """Predict if the face is real or spoof using the SVM model."""
    prediction = svm_model.predict(pca_features)
    return prediction[0]  # 1 for real, 0 for spoof
