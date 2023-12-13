'''
    TASK-3
    Implement a support vector machine (SVM) to classify images of cats and dogs from the given dataset.
'''

import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from skimage import io, transform
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure

# Function to read images and extract HOG features
def read_and_extract_features(folder_path, label):
    features = []
    labels = []

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = io.imread(image_path)
        image = transform.resize(image, (150, 150))  # Resize image for consistency

        # Converting image to grayscale
        gray_image = rgb2gray(image)

        # Extracting HOG features
        fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
        features.append(fd)
        labels.append(label)

    return features, labels

# Loading and preprocessing the dataset
cats_features, cats_labels = read_and_extract_features(r"D:/Internship/Prodigy InfoTech/training_set/cats", 0)
dogs_features, dogs_labels = read_and_extract_features(r"D:/Internship/Prodigy InfoTech/training_set/dogs", 1)

# Combining features and labels
X = cats_features + dogs_features
y = cats_labels + dogs_labels

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an SVM classifier
classifier = svm.SVC(kernel='linear')

# Training the classifier
classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluating the performance
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")












