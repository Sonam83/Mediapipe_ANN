import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# Model Path
# ==========================================================

MODEL_PATH = "models/face_landmarker.task"


# ==========================================================
# MediaPipe Classes
# ==========================================================

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode


# ==========================================================
# Image Mode Landmarker
# ==========================================================

image_options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.IMAGE,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False,

    num_faces=1

)

image_landmarker = FaceLandmarker.create_from_options(
    image_options
)


# ==========================================================
# Video Mode Landmarker
# ==========================================================

video_options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False,

    num_faces=1

)

video_landmarker = FaceLandmarker.create_from_options(
    video_options)


START_TIME = time.perf_counter()


# ==========================================================
# Feature Extraction
# ==========================================================

def _landmarks_to_features(result):

    if len(result.face_landmarks) == 0:
        return None

    landmarks = result.face_landmarks[0]

    features = []

    for point in landmarks:

        features.extend([

            point.x,

            point.y,

            point.z

        ])

    return np.array(features, dtype=np.float32)


# ==========================================================
# Upload Image
# ==========================================================

def extract_landmarks_image(image):

    rgb = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )

    result = image_landmarker.detect(
        mp_image
    )

    return _landmarks_to_features(result)


# ==========================================================
# Live Webcam
# ==========================================================

def extract_landmarks_video(frame):

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )

    timestamp = int(
        (time.perf_counter() - START_TIME) * 1000
    )

    result = video_landmarker.detect_for_video(

        mp_image,

        timestamp

    )

    return _landmarks_to_features(result)