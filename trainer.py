import cv2
import numpy as np
from PIL import Image
import os
import sys

def get_images_and_labels():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade")
        sys.exit(1)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    image_paths = []
    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("Error: No face images found in dataset directory")
        print("Please run face_capture.py first to create a dataset")
        sys.exit(1)
    
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        try:
            # Get ID from filename (User.1.1.jpg -> ID=1)
            id = int(os.path.split(image_path)[-1].split(".")[1])
            
            pil_img = Image.open(image_path).convert('L')  # Convert to grayscale
            img_np = np.array(pil_img, 'uint8')
            
            # Detect face in the image
            faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                print(f"Warning: No face detected in {image_path} - skipping")
                continue
                
            for (x, y, w, h) in faces:
                face_samples.append(img_np[y:y+h, x:x+w])
                ids.append(id)
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    if len(face_samples) == 0:
        print("Error: No valid face images found after processing")
        print("Possible reasons:")
        print("- No faces detected in your images")
        print("- Images are too low quality")
        print("- Incorrect file naming format (should be User.[id].[number].jpg)")
        sys.exit(1)
        
    return face_samples, ids

def train_model():
    print("\nStarting training process...")
    print("Looking for face images in dataset/ directory...")
    
    faces, ids = get_images_and_labels()
    
    print(f"Found {len(faces)} valid face samples")
    print(f"Number of unique persons: {len(np.unique(ids))}")
    
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    
    # Save the model
    recognizer.write('trainer/trainer.yml')
    print("\nTraining completed successfully!")
    print(f"Model saved to trainer/trainer.yml")
    print(f"Total persons trained: {len(np.unique(ids))}")

if __name__ == "__main__":
    train_model()