import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalize landmarks by subtracting the first landmark
    (same preprocessing used during training).

    Parameters
    ----------
    landmarks : numpy.ndarray
        Shape (1434,)

    Returns
    -------
    numpy.ndarray
        Shape (1434,)
    """

    landmarks = np.array(landmarks, dtype=np.float32)

    landmarks = landmarks.reshape(478, 3)

    reference = landmarks[0]

    landmarks = landmarks - reference

    return landmarks.flatten()