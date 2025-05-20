import cv2
import os
import time

def capture_faces():
    # Create directories
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get user info
    face_id = input("Enter user ID (numeric) and press <return> --> ")
    face_name = input("Enter user name and press <return> --> ")
    
    print(f"Capturing faces for {face_name}. Look at the camera...")
    
    count = 0
    while count < 300:  # Capture 300 samples
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            
            # Save the captured image
            cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
            
            # Display count
            cv2.putText(frame, f"Captured: {count}/300", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Capture', frame)
        
        if cv2.waitKey(100) & 0xff == 27:  # ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save name mapping
    with open('dataset/names.txt', 'a') as f:
        f.write(f"{face_id},{face_name}\n")
    
    print(f"Face capture completed for {face_name}")

if __name__ == "__main__":
    capture_faces()