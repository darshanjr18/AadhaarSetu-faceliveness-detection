import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

# Paths to the dataset folders
real_faces_path = 'C:/Users/priya/Downloads/Dataset 600 Images/Live'
spoof_faces_path = 'C:/Users/priya/Downloads/Dataset 600 Images/Spoof'

# Define image size and PCA parameters
image_size = (32, 32)
n_components = 50  # Number of principal components to keep

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized.flatten())
    return images

# Load images and labels
real_images = load_images_from_folder(real_faces_path)
spoof_images = load_images_from_folder(spoof_faces_path)

if len(real_images) == 0 or len(spoof_images) == 0:
    raise ValueError("No images found in the specified folders. Please check your paths.")

X = np.array(real_images + spoof_images)
y = np.array([1] * len(real_images) + [0] * len(spoof_images))  # 1 for real, 0 for spoof

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train SVM model
svm = SVC(kernel='linear', probability=True)  # You can experiment with different kernels
svm.fit(X_train_pca, y_train)

# Evaluate the model
accuracy = svm.score(X_test_pca, y_test)
print(f"SVM Model Accuracy: {accuracy:.2f}")

# Save the PCA, SVM models, and scaler
joblib.dump(pca, 'pca_model_compressed.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as well

print("PCA, SVM models, and scaler have been saved.")
