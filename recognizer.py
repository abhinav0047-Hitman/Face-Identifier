import cv2
import numpy as np
import os

def load_names():
    names = {}
    try:
        with open('dataset/names.txt', 'r') as f:
            for line in f:
                id, name = line.strip().split(',')
                names[int(id)] = name
    except FileNotFoundError:
        pass
    return names

def recognize_faces():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    
    names = load_names()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Try to recognize the face
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            if confidence < 100:
                name = names.get(id, f"Unknown {id}")
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "unknown"
                confidence_text = f"  {round(100 - confidence)}%"
            
            cv2.putText(frame, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(10) & 0xff == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()