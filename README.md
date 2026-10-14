# 😷 Face Mask Detection Using CNN

A real-time face mask detection application built with **PyTorch, a
Convolutional Neural Network (CNN), and OpenCV**. The application detects faces
from a webcam stream and classifies each detected face as `WithMask` or
`WithoutMask`.

The project includes a trained model, a webcam inference application, and a
Jupyter notebook for training and evaluation.

---

## 🚀 Overview

The application combines two computer vision steps:

1. **Face detection:** OpenCV's Haar Cascade locates faces in each webcam frame.
2. **Mask classification:** A trained PyTorch CNN classifies each face crop.

Predictions are shown on the live video with a bounding box and confidence
score. The application runs on CPU and automatically uses CUDA when available.

---

## ✨ Features

- 🎥 Real-time webcam capture and preview
- 🧑 Face detection using an OpenCV Haar Cascade
- 🧠 CNN-based `WithMask` / `WithoutMask` classification
- 📊 Prediction confidence displayed on each detected face
- ⚡ CPU inference with optional CUDA acceleration
- 🧪 Training, validation, and test workflow in a Jupyter notebook
- 📈 Training and validation accuracy/loss graph

---

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and training code |
| PyTorch | CNN architecture, training, and inference |
| Torchvision | Image transformations |
| OpenCV | Webcam capture, face detection, and preview |
| NumPy | Image and numerical operations |
| Jupyter Notebook | Model training and evaluation |
| KaggleHub | Dataset download in the training notebook |

---

## 📦 Project Dependencies

The application dependencies are listed in [`requirements.txt`](requirements.txt):

- `torch`
- `torchvision`
- `opencv-python`
- `numpy`

The training notebook additionally uses `kagglehub`, `pandas`, `matplotlib`,
and `scikit-learn`.

> **Note:** The webcam preview uses OpenCV GUI functions. Install
> `opencv-python`, not `opencv-python-headless`.

---

## 📂 Project Structure

```text
FaceMaskDetection/
├── docs/
│   └── training-metrics.png
├── model/
│   └── face_mask_cnn.pth
├── notebooks/
│   └── faceMaskTrainCNN.ipynb
├── src/
│   ├── detector.py
│   ├── faceDetector.py
│   └── model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

| File or directory | Description |
|---|---|
| `src/model.py` | Defines the CNN architecture |
| `src/faceDetector.py` | Loads the Haar Cascade face detector |
| `src/detector.py` | Runs live webcam inference |
| `model/face_mask_cnn.pth` | Trained model weights |
| `notebooks/faceMaskTrainCNN.ipynb` | Dataset download, training, and evaluation |
| `docs/training-metrics.png` | Accuracy and loss plots from the notebook |

---

## 🧠 Model Details

The custom CNN takes an RGB image of size **224 × 224** and produces scores for
two classes. Its feature extractor has five convolutional blocks with 16, 32,
64, 128, and 256 channels, followed by fully connected layers of 1024 and 512
units.

### Class Mapping

| Output index | Class |
|---:|---|
| 0 | `WithMask` |
| 1 | `WithoutMask` |

### Inference Preprocessing

The webcam face crop is converted from BGR to RGB, resized to 224 × 224, and
normalized with the values used in the notebook:

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

The class mapping and preprocessing must match the training configuration for
the checkpoint to produce meaningful predictions.

---

## 📊 Training and Evaluation

The notebook downloads the
[`ashishjangra27/face-mask-12k-images-dataset`](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
dataset using KaggleHub. It uses the `Train`, `Validation`, and `Test` folders
with PyTorch `ImageFolder`.

| Dataset split | Images |
|---|---:|
| Training | 10,000 |
| Validation | 800 |
| Test | 992 |

The recorded notebook results are **98.79% test accuracy** and `0.0793` test
loss. The confusion matrix is ordered `WithMask`, `WithoutMask` for both actual
rows and predicted columns:

| Actual \ Predicted | WithMask | WithoutMask |
|---|---:|---:|
| WithMask | 475 | 8 |
| WithoutMask | 4 | 505 |

### 📈 Training Graphs

Training and validation accuracy and loss across 10 epochs:

![Training vs validation loss and accuracy](docs/training-metrics.png)

These are results reported by the notebook on its test split. They do not
guarantee the same performance on live webcam images.

---

## ⚙️ Installation

### 1. Create a Virtual Environment

From the project root, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the Application Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell does not allow environment activation, use the virtual
environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## ▶️ Run the Application

From the project root:

```powershell
python .\src\detector.py
```

The application uses webcam device `0`. Press **Q** while the preview window is
focused to stop the application.

The model is loaded from `model/face_mask_cnn.pth`. The legacy filename
`model/mask_detector.pth` is also accepted. CUDA is used when available;
otherwise, inference runs on CPU.

---

## 📓 Run the Training Notebook

The notebook downloads the dataset from KaggleHub and therefore requires
internet access. Install its additional packages into the same environment:

```powershell
python -m pip install kagglehub pandas matplotlib scikit-learn
```

Open `notebooks/faceMaskTrainCNN.ipynb` in Jupyter or VS Code and run its cells.
The notebook trains the CNN, evaluates the test split, plots metrics, and saves
the weights as `face_mask_cnn.pth`. Place the resulting checkpoint in the
project's `model/` directory to use it with the webcam application.

---

## 🧭 Prediction Pipeline

For each video frame, the application:

1. Captures a frame from the webcam.
2. Converts it to grayscale for Haar Cascade face detection.
3. Crops each detected face and converts it to RGB.
4. Resizes and normalizes the crop.
5. Runs the CNN and converts its output to class probabilities.
6. Draws the predicted class and confidence on the frame.

---

## 🛠 Troubleshooting

### Webcam does not open

Close other applications that may be using the camera, check operating-system
camera permissions, and try another camera index in `src/detector.py`.

### `cv2.imshow()` is not implemented

Install `opencv-python` in the same environment used to run the app. The
headless OpenCV package does not support the preview window.

### Model checkpoint is missing or fails to load

Check that the checkpoint is located at `model/face_mask_cnn.pth` (or uses the
supported legacy name `model/mask_detector.pth`) and matches the architecture
in `src/model.py`.

### Predictions are inaccurate

Check lighting and face visibility. Confirm that the checkpoint uses the
documented class order and image normalization. Confidence is the model's
estimated probability, not a guarantee that a prediction is correct.

---

## ⚠️ Limitations

- Haar Cascade detection may be less reliable with poor lighting, unusual face
  angles, or partial occlusion.
- Test-set accuracy may not reflect performance on webcam images.
- Predictions depend on the training data and may not generalize to every mask,
  camera, or environment.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
The downloaded dataset is subject to its own terms.

---

## 👨‍💻 Developer

**Parth Borkar**
