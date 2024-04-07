from keras.models import load_model
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

model = load_model('emotion_detection_model.h5')

img_height, img_width = 224, 224

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        resized_face = cv2.resize(face_roi, (img_height, img_width))
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2RGB)
        
        preprocessed_face = rgb_face.astype('float32') / 255.0
        
        preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
        preprocessed_face = np.expand_dims(preprocessed_face, axis=-1)
        
        emotion_label = model.predict(preprocessed_face)[0]
        predicted_class_index = np.argmax(emotion_label)
        print(predicted_class_index,"predicted_class_index")
        
        detected_emotion = emotions[predicted_class_index]
        print(detected_emotion,"detected_emotion")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

