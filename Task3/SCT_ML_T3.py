import os
import cv2
import numpy as np
import urllib.request
import zipfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

def download_and_extract(max_images_per_class=1000):
    url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
    zip_path = 'kagglecatsanddogs.zip'
    extract_folder = 'PetImages'
    
    if not os.path.exists(extract_folder):
        if not os.path.exists(zip_path):
            print("Downloading dataset...")
            urllib.request.urlretrieve(url, zip_path)
            
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            cat_count, dog_count = 0, 0
            for file_info in zip_ref.infolist():
                if "PetImages/Cat/" in file_info.filename and file_info.filename.endswith(".jpg"):
                    if cat_count < max_images_per_class:
                        zip_ref.extract(file_info, '.')
                        cat_count += 1
                elif "PetImages/Dog/" in file_info.filename and file_info.filename.endswith(".jpg"):
                    if dog_count < max_images_per_class:
                        zip_ref.extract(file_info, '.')
                        dog_count += 1
                
                if cat_count >= max_images_per_class and dog_count >= max_images_per_class:
                    break
    return extract_folder

def load_images_with_hog(base_folder):
    X, y, images = [], [], []
    classes = {'Cat': 0, 'Dog': 1}
    
    for cls, label in classes.items():
        folder = os.path.join(base_folder, cls)
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                    X.append(features)
                    y.append(label)
                    images.append(img_resized)
            except Exception:
                pass
    return np.array(X), np.array(y), np.array(images)

print("Preparing dataset...")
extract_folder = download_and_extract(max_images_per_class=1000)

print("Extracting features...")
X, y, images = load_images_with_hog(extract_folder)

if len(X) > 0:
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, y, images, test_size=0.2, random_state=42)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    preds = svm.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=['Cat', 'Dog']))

    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_test[i], cmap='gray')
        pred_label = 'Cat' if preds[i] == 0 else 'Dog'
        true_label = 'Cat' if y_test[i] == 0 else 'Dog'
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("svm_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No images found.")
