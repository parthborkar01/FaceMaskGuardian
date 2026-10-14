import sys
from pathlib import Path

import cv2
import torch
from torchvision import transforms

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.model import CNN
    from src.faceDetector import load_face_detector
else:
    from .model import CNN
    from .faceDetector import load_face_detector


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CANDIDATES = [
    PROJECT_ROOT / "model" / "face_mask_cnn.pth",
    PROJECT_ROOT / "model" / "mask_detector.pth",
]
MODEL_PATH = next((candidate for candidate in MODEL_CANDIDATES if candidate.exists()), MODEL_CANDIDATES[0])


def load_model(device):
    model = CNN()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = load_model(device)
    print("Trained CNN loaded successfully!")
    print("Running on:", device)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    face_detector = load_face_detector()
    class_names = ["WithMask", "WithoutMask"]
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for (x, y, w, h) in faces:
                face = frame[y : y + h, x : x + w]

                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                image_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

                with torch.inference_mode():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_index = probabilities.max(dim=1)

                prediction = predicted_index.item()
                confidence = confidence.item()
                label = class_names[prediction]

                if label == "WithMask":
                    display_text = f"Mask Present: {confidence:.2%}"
                    color = (0, 255, 0)
                else:
                    display_text = f"Mask Absent: {confidence:.2%}"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    display_text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.imshow("Face Mask Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()