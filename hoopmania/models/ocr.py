"""Jersey number recognition using SmolVLM2."""

from typing import List, Optional
import numpy as np
import supervision as sv
from inference import get_model

from hoopmania.config import NUMBER_RECOGNITION_MODEL_ID


class JerseyOCR:
    """Jersey number recognition using SmolVLM2 from Roboflow.

    Fine-tuned model for reading jersey numbers from cropped images.
    """

    def __init__(
        self,
        model_id: str = NUMBER_RECOGNITION_MODEL_ID,
        prompt: str = "Read the number.",
    ):
        """Initialize the jersey OCR model.

        Args:
            model_id: Roboflow model ID
            prompt: Prompt to send to the VLM
        """
        self.model_id = model_id
        self.prompt = prompt
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = get_model(model_id=self.model_id)
        return self._model

    def recognize(self, crop: np.ndarray) -> str:
        """Recognize jersey number from a single crop.

        Args:
            crop: Cropped image of jersey number region

        Returns:
            Recognized number as string
        """
        # Resize to expected input size
        resized = sv.resize_image(crop, resolution_wh=(224, 224))
        result = self.model.predict(resized, self.prompt)[0]
        return result

    def recognize_batch(self, crops: List[np.ndarray]) -> List[str]:
        """Recognize jersey numbers from multiple crops.

        Args:
            crops: List of cropped images

        Returns:
            List of recognized numbers as strings
        """
        return [self.recognize(crop) for crop in crops]

    @staticmethod
    def prepare_crops(
        frame: np.ndarray,
        detections: sv.Detections,
        padding: int = 10,
    ) -> List[np.ndarray]:
        """Prepare number crops from detections.

        Args:
            frame: BGR image as numpy array
            detections: Number detections
            padding: Padding to add around each detection

        Returns:
            List of padded and cropped images
        """
        frame_h, frame_w = frame.shape[:2]

        # Pad boxes and clip to frame bounds
        padded_boxes = sv.pad_boxes(xyxy=detections.xyxy, px=padding, py=padding)
        clipped_boxes = sv.clip_boxes(padded_boxes, (frame_w, frame_h))

        return [sv.crop_image(frame, xyxy) for xyxy in clipped_boxes]
