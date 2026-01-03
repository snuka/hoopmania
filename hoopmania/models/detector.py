"""Player detection using RF-DETR model from Roboflow."""

from typing import Optional
import numpy as np
import supervision as sv
from inference import get_model

from hoopmania.config import (
    PLAYER_DETECTION_MODEL_ID,
    PLAYER_DETECTION_CONFIDENCE,
    PLAYER_DETECTION_IOU_THRESHOLD,
    PLAYER_CLASS_IDS,
    NUMBER_CLASS_ID,
    JUMP_SHOT_CLASS_ID,
    LAYUP_DUNK_CLASS_ID,
    BALL_IN_BASKET_CLASS_ID,
)


class PlayerDetector:
    """RF-DETR based player detector for basketball videos."""

    def __init__(
        self,
        model_id: str = PLAYER_DETECTION_MODEL_ID,
        confidence: float = PLAYER_DETECTION_CONFIDENCE,
        iou_threshold: float = PLAYER_DETECTION_IOU_THRESHOLD,
    ):
        """Initialize the player detector.

        Args:
            model_id: Roboflow model ID
            confidence: Detection confidence threshold
            iou_threshold: NMS IOU threshold
        """
        self.model_id = model_id
        self.confidence = confidence
        self.iou_threshold = iou_threshold
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
        iou_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Detect all objects in a frame.

        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold
            iou_threshold: Override default IOU threshold

        Returns:
            sv.Detections with all detected objects
        """
        conf = confidence or self.confidence
        iou = iou_threshold or self.iou_threshold

        result = self.model.infer(frame, confidence=conf, iou_threshold=iou)[0]
        return sv.Detections.from_inference(result)

    def detect_players(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Detect only player-related objects (players, possessions, shots, blocks).

        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold
            iou_threshold: Override default IOU threshold

        Returns:
            sv.Detections filtered to player classes only
        """
        detections = self.detect(frame, confidence, iou_threshold)
        return detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]

    def detect_numbers(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> sv.Detections:
        """Detect jersey number regions.

        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold
            iou_threshold: Override default IOU threshold

        Returns:
            sv.Detections filtered to number class only
        """
        detections = self.detect(frame, confidence, iou_threshold)
        return detections[detections.class_id == NUMBER_CLASS_ID]

    def detect_shot_events(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> dict:
        """Detect shot-related events.

        Args:
            frame: BGR image as numpy array
            confidence: Override default confidence threshold
            iou_threshold: Override default IOU threshold

        Returns:
            Dictionary with boolean flags for each shot event type
        """
        detections = self.detect(frame, confidence, iou_threshold)

        return {
            "has_jump_shot": len(detections[detections.class_id == JUMP_SHOT_CLASS_ID]) > 0,
            "has_layup_dunk": len(detections[detections.class_id == LAYUP_DUNK_CLASS_ID]) > 0,
            "has_ball_in_basket": len(detections[detections.class_id == BALL_IN_BASKET_CLASS_ID]) > 0,
        }
