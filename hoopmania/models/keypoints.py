"""Court keypoint detection for homography estimation."""

from typing import Optional, Tuple
import numpy as np
import supervision as sv
from inference import get_model

from hoopmania.config import (
    KEYPOINT_DETECTION_MODEL_ID,
    KEYPOINT_DETECTION_CONFIDENCE,
    KEYPOINT_ANCHOR_CONFIDENCE,
)


class CourtKeypointDetector:
    """Court keypoint detector for basketball court homography.

    Detects court landmarks that can be used to compute
    homography matrix for mapping player positions to 2D court.
    """

    def __init__(
        self,
        model_id: str = KEYPOINT_DETECTION_MODEL_ID,
        confidence: float = KEYPOINT_DETECTION_CONFIDENCE,
        anchor_confidence: float = KEYPOINT_ANCHOR_CONFIDENCE,
    ):
        """Initialize the keypoint detector.

        Args:
            model_id: Roboflow model ID
            confidence: Detection confidence threshold
            anchor_confidence: Higher threshold for anchor points
        """
        self.model_id = model_id
        self.confidence = confidence
        self.anchor_confidence = anchor_confidence
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = get_model(model_id=self.model_id)
        return self._model

    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
    ) -> sv.KeyPoints:
        """Detect all court keypoints.

        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold

        Returns:
            sv.KeyPoints with detected court landmarks
        """
        conf = confidence or self.confidence
        result = self.model.infer(frame, confidence=conf)[0]
        return sv.KeyPoints.from_inference(result)

    def detect_anchors(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        anchor_confidence: Optional[float] = None,
    ) -> sv.KeyPoints:
        """Detect high-confidence anchor keypoints.

        These are used for more reliable homography estimation.

        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold
            anchor_confidence: Override anchor confidence threshold

        Returns:
            sv.KeyPoints filtered to high-confidence points
        """
        key_points = self.detect(frame, confidence)
        anchor_conf = anchor_confidence or self.anchor_confidence

        # Filter to high confidence points
        mask = key_points.confidence[0] > anchor_conf
        return key_points[:, mask]

    def get_landmarks_for_homography(
        self,
        frame: np.ndarray,
        min_points: int = 4,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get keypoint pairs suitable for homography computation.

        Args:
            frame: BGR image as numpy array
            min_points: Minimum number of points required

        Returns:
            Tuple of (frame_landmarks, mask) where mask indicates
            which court vertices have corresponding frame points.
            Returns (None, None) if insufficient points.
        """
        key_points = self.detect(frame)

        if len(key_points) == 0:
            return None, None

        # Get high confidence mask
        mask = key_points.confidence[0] > self.anchor_confidence

        if np.count_nonzero(mask) < min_points:
            return None, None

        frame_landmarks = key_points[:, mask].xy[0]
        return frame_landmarks, mask
