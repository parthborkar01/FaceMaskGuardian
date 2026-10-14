from pathlib import Path

import cv2


def load_face_detector():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "OpenCV Haar cascade support is missing. Install the project requirements."
        )

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))

    if face_detector.empty():
        raise RuntimeError(f"Could not load Haar Cascade from {cascade_path}")

    return face_detector