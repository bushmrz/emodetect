import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained model
model = load_model('fer_model.h5')

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Surprise']


def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face


def predict_emotion(face):
    predictions = model.predict(face)
    max_index = np.argmax(predictions[0])
    return emotion_labels[max_index], predictions[0][max_index] * 100


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            preprocessed_face = preprocess_face(face)
            emotion, confidence = predict_emotion(preprocessed_face)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Display the emotion and confidence
            cv2.putText(frame, f'{emotion} ({confidence:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
