# 😎 Face Identifier System (Python OpenCV Project)

A simple yet effective Python-based facial recognition system using OpenCV. This project allows you to **capture faces**, **train a recognition model**, and **identify individuals** in real-time using a webcam.

---

## 📂 Project Files

| File Name         | Purpose                                                      |
|------------------|--------------------------------------------------------------|
| `Face_capture.py` | Capture face images and store them in a dataset folder.     |
| `trainer.py`      | Train an LBPH face recognizer using collected face images.  |
| `recognizer.py`   | Run real-time face recognition using the trained model.     |

---

## 🧠 Features

- 📸 Capture and label multiple face samples
- 🏋️ Train with OpenCV's **LBPH Face Recognizer**
- 👁️ Real-time face detection and recognition
- 💾 Face data stored locally for simplicity
- 🔍 Annotated face bounding boxes with name labels

---

## 🛠️ Technology Used

- **Python 3.7+**
- **OpenCV** (cv2)
- **NumPy**
- **OS / Pickle** (for saving and loading labels)

---

## 🖥️ How It Works

### 1. `Face_capture.py`
Captures 300 grayscale images of a person’s face from the webcam and stores them in `dataset/` with a label.

### 2. 'trainer.py' 
Trains the model according to the data set.

### 3. 'Face_Capture.py 
Real time identify the faces.
