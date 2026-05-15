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

def download_and_extract(max_images_per_class=200):
    url = 'https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip'
    zip_path = 'rockpaperscissors.zip'
    extract_folder = 'rockpaperscissors'
    
    if not os.path.exists(extract_folder):
        if not os.path.exists(zip_path):
            print("Downloading Rock Paper Scissors hand gesture dataset (~300MB)...")
            urllib.request.urlretrieve(url, zip_path)
            
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            counts = {'rock': 0, 'paper': 0, 'scissors': 0}
            for file_info in zip_ref.infolist():
                filename = file_info.filename.lower()
                

                gesture_class = None
                for g in counts.keys():
                    if f"/{g}/" in filename and filename.endswith(".png"):
                        gesture_class = g
                        break
                        
                if gesture_class and counts[gesture_class] < max_images_per_class:
                    zip_ref.extract(file_info, '.')
                    counts[gesture_class] += 1
                    
                if all(c >= max_images_per_class for c in counts.values()):
                    break
                    
    return os.path.join(extract_folder, 'rps-cv-images') if os.path.exists(os.path.join(extract_folder, 'rps-cv-images')) else extract_folder

def load_hand_gestures(base_folder, max_per_class=200):
    X, y, images = [], [], []
    gesture_labels = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for gesture_name, label in gesture_labels.items():
     
        target_dir = None
        for root, dirs, files in os.walk(base_folder):
            if gesture_name in dirs:
                target_dir = os.path.join(root, gesture_name)
                break
                
        if not target_dir:
            continue
            
        count = 0
        for filename in os.listdir(target_dir):
            if count >= max_per_class: break
            try:
                img_path = os.path.join(target_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (64, 64))
                    features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
                    X.append(features)
                    y.append(label)
                    images.append(img_resized)
                    count += 1
            except Exception:
                pass
                
    target_names = [name for name, _ in sorted(gesture_labels.items(), key=lambda item: item[1])]
    return np.array(X), np.array(y), np.array(images), target_names

print("Preparing real hand gesture dataset...")
extract_folder = download_and_extract(max_images_per_class=200)

print("Loading hand gesture images...")
X, y, images, target_names = load_hand_gestures(extract_folder, max_per_class=200)

if len(X) > 0:
    print(f"Loaded {len(X)} real images across {len(target_names)} gestures.")
    print("Splitting dataset and training SVM...")
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(X, y, images, test_size=0.2, random_state=42)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    preds = svm.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=target_names))

    print("Generating prediction visualization...")
    plt.figure(figsize=(12, 6))
    for i in range(min(10, len(img_test))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_test[i], cmap='gray')
        pred_label = target_names[preds[i]]
        true_label = target_names[y_test[i]]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("gesture_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No images processed. Something went wrong with the dataset download.")
