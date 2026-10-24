# Face Recognition System using Deep CNN Embeddings & Webcam

An end-to-end Face Recognition System built with Python, OpenCV, PyTorch, and FaceNet (`Inception-ResNet-v1`). The system captures face images from a live webcam, crops and normalizes them, extracts 512-dimensional facial embeddings using a pre-trained deep convolutional neural network, and identifies registered individuals in real time while tagging unregistered people as **"Unknown"**.

---

## 📌 Project Tasks & Mapping

This project fulfills all 10 core computer vision and machine learning tasks:

| Step / Script | Corresponding Tasks | Description |
| :--- | :--- | :--- |
| `01_collect_data.py` | **Tasks 1 & 2** | Captures face photos via webcam (with auto-mirroring & auto-append support) and organizes them by identity under `dataset/`. |
| `02_preprocess_and_split.py` | **Tasks 3, 4, 5** | Detects and crops faces, resizes to 160×160, normalizes pixel values to `[-1, 1]`, and splits data into 80% train and 20% test. |
| `03_train_model.py` | **Tasks 6 & 7** | Extracts 512-D deep feature embeddings using pre-trained FaceNet and builds a multi-sample vector database in `models/face_model.pkl`. |
| `04_evaluate_model.py` | **Tasks 8 & 9** | Tests model against known and unknown faces, calculating Accuracy, Precision, Recall, F1-Score, and saving a Confusion Matrix plot. |
| `05_live_recognition.py` | **Task 10** | Live webcam inference: draws green boxes for registered persons and red boxes for unknown visitors. |

---

## 📁 Directory Structure

```text
Face Recognition System/
│
├── dataset/                        # Raw face images captured per user
│   ├── sourasish/                  # Captured photos for user 1
│   └── person_2/                   # Captured photos for user 2
│
├── dataset_split/                  # Preprocessed 160x160 normalized images
│   ├── train/                      # 80% training set
│   └── test/                       # 20% testing set
│
├── models/
│   ├── face_model.pkl              # Saved 512-D embeddings vector database
│   └── confusion_matrix.png        # Evaluation metric plot
│
├── 01_collect_data.py              # Task 1 & 2: Dataset collection
├── 02_preprocess_and_split.py      # Task 3, 4, 5: Preprocessing & split
├── 03_train_model.py               # Task 6 & 7: Feature extraction & training
├── 04_evaluate_model.py            # Task 8 & 9: Evaluation & metrics
├── 05_live_recognition.py          # Task 10: Real-time webcam identification
│
├── requirements.txt                # Pinned compatible dependencies
├── .gitignore                      # Keeps personal photos and cache out of Git
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Key Features & Engineering Highlights

* **Beginner-Friendly & Report-Ready**: Clean, linear Python scripts without complex nested classes or ternary operators, making them easy to document in college project files and explain during viva examinations.
* **Natural Mirror View**: Automatically flips the webcam feed horizontally (`cv2.flip(frame, 1)`) so it behaves like a natural mirror or smartphone selfie camera.
* **Spectacles & Multi-Condition Support**:
  * **Boundary Padding (`pad = 20`)**: Preserves full face geometry and spectacle frames without clipping.
  * **Auto-Append Mode**: Running `01_collect_data.py` again appends new samples without overwriting previous photos. You can easily register 30 photos without glasses and 30 photos with glasses.
  * **Nearest-Sample Vector Matching**: Compares live faces against all individual condition vectors of the user, achieving high recognition confidence whether you wear glasses, change hairstyles, or adjust lighting.
* **Rejecting Unknown Strangers**: Uses an empirical Cosine Similarity threshold ($0.65$). Any unregistered face is automatically labeled **"Unknown"** with a red bounding box.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/face-recognition-system.git
cd face-recognition-system
```

### 2. Create and activate a virtual environment
```powershell
# On Windows PowerShell:
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📖 Step-by-Step Execution Guide

### Step 1: Collect Face Dataset (Tasks 1 & 2)
```bash
python 01_collect_data.py
```
* Enter your name when prompted (e.g. `sourasish`).
* Look into the camera while it captures 30 photos.
* *(Tip for glasses wearers)*: Run the script a second time with your glasses on! The script will automatically add 30 more photos, giving you 60 photos total (30 without glasses + 30 with glasses).
* You can also repeat this step for friends or family members to register multiple users.

### Step 2: Preprocess and Split the Dataset (Tasks 3, 4 & 5)
```bash
python 02_preprocess_and_split.py
```
* Crops faces and standardizes them to $160 \times 160$ pixels.
* Automatically splits each person's photos into an 80% **training** set and a 20% **testing** set.

### Step 3: Extract CNN Features & Train Model (Tasks 6 & 7)
```bash
python 03_train_model.py
```
* Passes face images through pre-trained FaceNet (`InceptionResnetV1` trained on VGGFace2).
* Extracts $L_2$-normalized 512-dimensional embeddings.
* Saves the multi-sample vector database to `models/face_model.pkl`.

### Step 4: Evaluate Model Performance (Tasks 8 & 9)
```bash
python 04_evaluate_model.py
```
* Tests the model against the test set.
* Generates a full classification report (**Accuracy, Precision, Recall, F1-Score**).
* Plots and saves a confusion matrix heatmap to `models/confusion_matrix.png`.

### Step 5: Test Real-Time Webcam Recognition (Task 10)
```bash
python 05_live_recognition.py
```
* Launches the live webcam feed.
* **Green Box**: Shows the recognized person's name and match confidence score (e.g. `sourasish (88%)`).
* **Red Box**: Shows `Unknown (42%)` when an unregistered person enters the frame.
* Press **`q`** on the video window to close it.

---

## 🧠 Theory & Concepts

### 1. Face Detection vs. Face Recognition
* **Face Detection** (OpenCV Haar Cascade): Solves the spatial localization problem — *"Where is the face located in this frame?"*
* **Face Recognition** (FaceNet CNN): Solves the identity classification problem — *"Whose face is this?"*

### 2. Deep Facial Embeddings
Rather than comparing raw pixels (which fail when lighting or pose changes), FaceNet maps face images into a 512-dimensional vector space:
$$\text{Similarity}(e_{\text{live}}, e_{\text{db}}) = \frac{e_{\text{live}} \cdot e_{\text{db}}}{\|e_{\text{live}}\|_2 \|e_{\text{db}}\|_2} = e_{\text{live}} \cdot e_{\text{db}} \quad (\text{since embeddings are } L_2 \text{ normalized})$$

* Faces of the **same individual** produce a cosine similarity $\ge 0.65$.
* Faces of **different individuals** produce a cosine similarity $< 0.50$.

### 3. Unknown Person Rejection
$$\text{Decision} = \begin{cases} \text{Person Name} & \text{if } \max(\text{Similarity}) \ge 0.65 \\ \text{"Unknown"} & \text{if } \max(\text{Similarity}) < 0.65 \end{cases}$$

---

## 📤 Pushing to GitHub

To publish this project to your GitHub account:

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

---

## 📤 Pushing to GitHub

To publish this project to your GitHub account:

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

