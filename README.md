# Face Mask Detection Using Deep Learning

A real-time face mask detection application built with **PyTorch, Convolutional Neural Networks (CNNs), and OpenCV**. The application detects faces from a webcam stream and classifies each detected face as `WithMask` or `WithoutMask` using a trained deep learning model.

The model is trained in Google Colab, and the trained checkpoint is deployed locally for real-time inference on CPU or GPU.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training and Evaluation](#training-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Prediction Pipeline](#prediction-pipeline)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

This project demonstrates the application of deep learning to real-time computer vision.

The system combines two components:

1. **Face Detection:** OpenCV's Haar Cascade classifier locates faces in webcam frames.
2. **Mask Classification:** A custom CNN trained with PyTorch classifies each detected face into one of two categories:
   - `WithMask`
   - `WithoutMask`

The application displays a bounding box, predicted class, and confidence score around each detected face.

The model is trained using a face mask image dataset in Google Colab. Once training is complete, the model weights are downloaded and used for inference on a local machine.

## Features

- Real-time face detection through a webcam.
- Custom CNN architecture implemented using PyTorch.
- Binary classification of masked and unmasked faces.
- Face detection using OpenCV Haar Cascades.
- Confidence scores displayed alongside predictions.
- CPU inference support for systems without a dedicated GPU.
- Optional CUDA acceleration when compatible hardware is available.
- Separate training and inference workflows.
- Model evaluation using training, validation, and test datasets.

---

## System Architecture

```text
                Webcam
                   |
                   v
           OpenCV Video Capture
                   |
                   v
            Frame Preprocessing
                   |
                   v
          Haar Cascade Face Detection
                   |
                   v
             Face Extraction
                   |
                   v
         Image Resize and Normalization
                   |
                   v
             Trained CNN Model
                   |
                   v
           Mask Classification
                   |
            +------+------+
            |             |
            v             v
         WithMask     WithoutMask
            |             |
            +------+------+
                   |
                   v
       Bounding Box and Confidence
                   |
                   v
            Display Webcam Frame
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | CNN implementation and model inference |
| OpenCV | Face detection, webcam capture, and visualization |
| Torchvision | Image preprocessing and transformations |
| NumPy | Numerical operations and image processing |
| Google Colab | Model training and experimentation |

---

## Project Structure

```text
FaceMaskDetection/
│
├── model/
│   └── face_mask_cnn.pth
│
├── notebooks/
│   └── faceMaskTrainCNN.ipynb
│
├── src/
│   ├── detector.py
│   ├── faceDetector.py
│   └── model.py
│
├── docs/
│   └── training-metrics.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Directory Description

| Path | Description |
|---|---|
| `model/` | Contains the trained CNN checkpoint. |
| `notebooks/` | Contains the Google Colab training notebook. |
| `src/model.py` | Defines the CNN architecture. |
| `src/faceDetector.py` | Handles face detection. |
| `src/detector.py` | Runs real-time webcam inference. |
| `docs/` | Contains training graphs and project documentation. |
| `requirements.txt` | Lists the required Python packages. |

---

## Model Details

The project uses a custom Convolutional Neural Network implemented in PyTorch.

The CNN learns visual features from labeled images and uses them to classify detected faces.

### Classification Classes

| Output Index | Class | Description |
|---:|---|---|
| 0 | `WithMask` | The model predicts that a mask is present. |
| 1 | `WithoutMask` | The model predicts that a mask is absent. |

The class mapping must match the mapping used during training.

### Input Configuration

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 pixels |
| Input channels | 3 (RGB) |
| Number of classes | 2 |
| Input normalization mean | `[0.485, 0.456, 0.406]` |
| Input normalization standard deviation | `[0.229, 0.224, 0.225]` |

The preprocessing configuration above must match the transformations used during training.

### Model Checkpoint

The trained weights are stored in:

```text
model/face_mask_cnn.pth
```

The checkpoint contains the learned model parameters. The CNN architecture is defined separately in the source code.

---

## Training and Evaluation

Model training is performed in Google Colab using PyTorch.

The training workflow includes:

1. Loading the face mask dataset.
2. Preprocessing and transforming images.
3. Creating training, validation, and test datasets.
4. Initializing the custom CNN.
5. Training the model using labeled images.
6. Evaluating performance on validation and test datasets.
7. Saving the trained model weights for local deployment.

### Reported Test Performance

The training notebook reports **98.79% test accuracy** on a test set containing 992 images.

The following metrics were recorded during a 10-epoch training run:

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 94.22% | 97.12% |
| 2 | 97.45% | 97.38% |
| 3 | 98.11% | 99.12% |
| 4 | 98.63% | 98.88% |
| 5 | 98.85% | 99.38% |
| 6 | 98.57% | 98.50% |
| 7 | 98.82% | 99.75% |
| 8 | 99.10% | 99.50% |
| 9 | 99.43% | 99.62% |
| 10 | 98.96% | 99.00% |

> **Note:** These are the metrics reported by the training notebook. Test accuracy alone does not establish how reliably the model will perform on real-world webcam images. Performance also depends on dataset quality, class distribution, preprocessing, and test-set independence.

### Training Visualization

The following image contains the training and validation accuracy and loss curves.

![Training and validation accuracy and loss](docs/training-metrics.png)

---

## Installation

### Prerequisites

- Python 3.10 or newer.
- A working webcam.
- Git (optional).
- A compatible CPU or CUDA-enabled GPU.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd FaceMaskDetection
```

Replace `<your-repository-url>` with your GitHub repository URL.

Alternatively, download the repository and open its directory in VS Code.

### 2. Create a Virtual Environment

On Windows:

```powershell
py -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the packages required by the application, including PyTorch, Torchvision, and `opencv-python`.

**Important:** Use the GUI-enabled `opencv-python` package. Do not install `opencv-python-headless` for this application because the webcam preview uses OpenCV GUI functions.

### 4. Verify the Installation

Run:

```python
import cv2
import torch
import torchvision

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## Usage

### 1. Verify the Model Checkpoint

Ensure that the trained checkpoint is located at:

```text
model/face_mask_cnn.pth
```

If your application supports the legacy filename, you can also use:

```text
model/mask_detector.pth
```

The checkpoint architecture, class mapping, and preprocessing must match those used during training.

### 2. Start the Application

From the project root, run:

```powershell
python .\src\detector.py
```

The application attempts to open webcam device `0`.

### 3. Run Real-Time Detection

Once the webcam window opens:

- Position your face within the camera view.
- Allow the application to detect your face.
- Observe the predicted mask class and confidence score.
- Press **Q** while the preview window is focused to stop the application.

The application releases the webcam and closes the OpenCV windows when it exits.

---

## Prediction Pipeline

For each webcam frame, the application performs the following operations:

1. Capture the frame using OpenCV.
2. Convert the frame to grayscale for Haar Cascade face detection.
3. Detect face locations.
4. Crop each detected face.
5. Convert the crop from BGR to RGB.
6. Resize the image to 224 × 224 pixels.
7. Apply the same normalization used during training.
8. Convert the image into a PyTorch tensor.
9. Run inference using the trained CNN.
10. Convert model outputs into class probabilities.
11. Draw a bounding box, predicted class, and confidence score.

The model uses the predicted class to display either `WithMask` or `WithoutMask`.

---

## Troubleshooting

### Could Not Open Webcam

Possible causes:

- Another application is using the camera.
- Camera permissions are disabled.
- The webcam is disconnected.
- The selected camera index is incorrect.

Try closing other camera applications, checking system permissions, and confirming that the webcam is available.

### Could Not Read Webcam Frame

Verify that the camera is enabled and functioning correctly. Restart the application after checking the camera connection.

### `cv2.imshow()` Is Not Implemented

Install the GUI-enabled OpenCV package in the same Python environment used to run the application:

```bash
pip uninstall opencv-python-headless
pip install opencv-python
```

### Model Checkpoint Not Found

Confirm that the checkpoint exists at the expected path:

```text
model/face_mask_cnn.pth
```

Check the filename and run the application from the project root.

### Model Loading Error

A state-dictionary loading error may indicate that:

- The CNN architecture differs from the training architecture.
- The checkpoint contains a different object or dictionary structure.
- The model layer dimensions do not match.
- The checkpoint was saved using a different model definition.

Use the same CNN architecture that was used to train the checkpoint.

### Predictions Are Inaccurate

Check the following:

- Input dimensions match the training configuration.
- RGB/BGR conversion is correct.
- Image normalization matches the training transformations.
- The class-to-index mapping is correct.
- The webcam provides sufficient lighting.
- Faces are reasonably visible and positioned within the frame.

A confidence score represents the model's output confidence; it does not guarantee that a prediction is correct.

---

## Limitations

- Haar Cascade may miss faces under challenging lighting, pose, or occlusion conditions.
- Classification quality depends on the diversity and quality of the training dataset.
- A model trained on ordinary masked and unmasked faces may confuse masks with other objects covering the face.
- High test accuracy does not guarantee equivalent real-world performance.
- The application classifies detected faces; it does not independently verify the material or authenticity of a face covering.

---

## Future Improvements

Potential extensions include:

- Evaluate false-positive and false-negative rates.
- Add more diverse examples of face occlusion to the training dataset.
- Compare Haar Cascade with modern face detection models.
- Optimize inference speed for CPU-based systems.
- Add a confidence threshold and an uncertain prediction state.
- Develop a simple graphical user interface.
- Export the model to an optimized inference format.
- Evaluate robustness under different lighting conditions, face orientations, and camera qualities.

---

## License

Add the license that applies to your project and verify that it is compatible with the licenses of the datasets and third-party components you use.

---

## Author

**Your Name**

Face Mask Detection using PyTorch, CNN, and OpenCV.