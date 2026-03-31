import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "dataset/train"
IMAGE_SIZE = (128, 128)
MAX_IMAGES = 3000

def load_data(path):
    data = []
    labels = []

    files = os.listdir(path)
    np.random.shuffle(files)

    for i, img_name in enumerate(files):
        if i >= MAX_IMAGES:
            break

        try:
            if not (img_name.lower().startswith("cat") or img_name.lower().startswith("dog")):
                continue

            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            
            image = cv2.resize(image, IMAGE_SIZE)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

           
            features = hog(
                gray,
                orientations=12,
                pixels_per_cell=(8, 8),
                cells_per_block=(3, 3),
                block_norm='L2-Hys'
            )

            data.append(features)

            if img_name.lower().startswith("cat"):
                labels.append(0)
            else:
                labels.append(1)

        except:
            continue

    return np.array(data), np.array(labels)


def main():
    print(" Loading data...")
    X, y = load_data(DATASET_PATH)

    print(" Samples:", len(X))
    print(" Distribution:", np.bincount(y))

    if len(np.unique(y)) < 2:
        print(" Error: Need at least 2 classes")
        return

    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    print(" Applying PCA...")
    pca = PCA(n_components=150)
    X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(" Training SVM...")
    model = SVC(kernel='rbf', C=100, gamma=0.001)
    model.fit(X_train, y_train)


    print(" Evaluating...")
    y_pred = model.predict(X_test)

    print("\n Accuracy:", round(accuracy_score(y_test, y_pred), 2))

    print("\n Unique Predictions:", np.unique(y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()