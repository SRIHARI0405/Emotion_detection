import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
dataset_loc = 'dataset/train'
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
min_images_per_category = 1000

#preprocessing the images
img_height, img_width = 224, 224
X_train = []
y_train = []
images_processed = {emotion: 0 for emotion in emotions}

for i, emotion in enumerate(emotions):
    emotion_dir = os.path.join(dataset_loc, emotion)
    for img_name in os.listdir(emotion_dir):
        if images_processed[emotion] >= min_images_per_category:
            break  
        img_path = os.path.join(emotion_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            X_train.append(cv2.resize(img, (img_height, img_width)))
            y_train.append(i)
            images_processed[emotion] += 1

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.astype('float32') / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=27)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

#Training History
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#Confusion Matrix
y_pred = model.predict(X_val)
cm = confusion_matrix(y_val, np.argmax(y_pred, axis=1))
print(cm)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#model save
model.save('emotion_detection_model.h5')







