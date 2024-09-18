import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib


# Function to load and preprocess face images from subfolders
def load_images_from_subfolders(root_folder, image_size=(32, 32)):
    images = []
    for person_folder in os.listdir(root_folder):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):  # Ensure it's a directory
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                if os.path.isfile(img_path):  # Ensure it's a file and not a directory
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                    if img is not None:
                        img_resized = cv2.resize(img, image_size)  # Resize the image to 32x32
                        img_flattened = img_resized.flatten()  # Flatten the image (convert 2D image to 1D array)
                        images.append(img_flattened)
                    else:
                        print(f"Failed to read image: {img_path}")
                else:
                    print(f"Skipping non-file: {img_path}")
        else:
            print(f"Skipping non-directory: {person_path}")

    if not images:
        print("No images were loaded. Please check the dataset folder.")
    return np.array(images)


# Use your dataset path
dataset_folder = 'C:/Users/priya/Downloads/archive (2)/lfw-deepfunneled/lfw-deepfunneled'

# Load the images from the dataset
X = load_images_from_subfolders(dataset_folder)

# Debugging: Print shape of X
print(f"Shape of X before PCA: {X.shape}")

if X.shape[0] == 0:
    print("No images were loaded. Please check the dataset path and ensure the folder contains valid image files.")
else:
    # PCA Optimization: Reduce the number of components (eigenfaces) to reduce model size
    n_components = 50  # Adjust this as necessary

    pca = PCA(n_components=n_components)

    # Fit PCA to the dataset
    try:
        X_pca = pca.fit_transform(X)
        print(f"Shape of X_pca after PCA: {X_pca.shape}")
    except ValueError as e:
        print(f"Error during PCA fitting: {e}")

    # Optional: Train a Linear SVM on the PCA-transformed data (if you have labels)
    # y = np.array([...])  # Replace with actual labels if available
    # svm = SVC(kernel='linear')  # Linear SVM
    # svm.fit(X_pca, y)

    # Save PCA model with compression
    joblib.dump(pca, 'pca_model_compressed.pkl', compress=3)

    print("PCA training completed and model saved with compression.")



