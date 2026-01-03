"""SAM2 based player tracking."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import supervision as sv

from hoopmania.config import SAM2_DIR, SAM2_CHECKPOINT, SAM2_CONFIG


class SAM2Tracker:
    """SAM2 based player tracker for basketball videos.

    Uses segment-anything-2-real-time for real-time player tracking with masks.
    """

    def __init__(
        self,
        checkpoint: Optional[Path] = None,
        config: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the SAM2 tracker.

        Args:
            checkpoint: Path to SAM2 checkpoint file
            config: SAM2 config name
            device: Device to run inference on (auto-detected if None)
        """
        self.checkpoint = checkpoint or SAM2_CHECKPOINT
        self.config = config or SAM2_CONFIG

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._predictor = None
        self._prompted = False

    def _ensure_sam2_imported(self):
        """Ensure SAM2 is importable by adding to path."""
        sam2_path = str(SAM2_DIR)
        if sam2_path not in sys.path:
            sys.path.insert(0, sam2_path)

    @property
    def predictor(self):
        """Lazy load the SAM2 predictor."""
        if self._predictor is None:
            self._ensure_sam2_imported()
            from sam2.build_sam import build_sam2_camera_predictor

            self._predictor = build_sam2_camera_predictor(
                str(self.config),
                str(self.checkpoint)
            )
        return self._predictor

    def prompt_first_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> None:
        """Initialize tracking with detections from first frame.

        Args:
            frame: BGR image as numpy array
            detections: Initial player detections with bounding boxes

        Raises:
            ValueError: If no detections provided
        """
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")

        # Assign tracker IDs if not present
        if detections.tracker_id is None:
            detections.tracker_id = np.arange(1, len(detections) + 1)

        with torch.inference_mode():
            # Use autocast only for CUDA (MPS doesn't support bfloat16 autocast)
            if self.device == "cuda":
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    self._prompt_first_frame_impl(frame, detections)
            else:
                self._prompt_first_frame_impl(frame, detections)

        self._prompted = True

    def _prompt_first_frame_impl(self, frame: np.ndarray, detections: sv.Detections) -> None:
        """Internal implementation of first frame prompting."""
        self.predictor.load_first_frame(frame)
        for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
            bbox = np.asarray([xyxy], dtype=np.float32)
            self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=int(obj_id),
                bbox=bbox,
            )

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        """Track objects to next frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            sv.Detections with masks and tracker IDs

        Raises:
            RuntimeError: If prompt_first_frame not called
        """
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        with torch.inference_mode():
            # Use autocast only for CUDA (MPS doesn't support bfloat16 autocast)
            if self.device == "cuda":
                with torch.autocast(self.device, dtype=torch.bfloat16):
                    tracker_ids, mask_logits = self.predictor.track(frame)
            else:
                tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        if masks.ndim == 2:
            masks = masks[None, ...]

        # Filter segments by distance to remove noise
        masks = np.array([
            sv.filter_segments_by_distance(mask, relative_distance=0.03, mode="edge")
            for mask in masks
        ])

        xyxy = sv.mask_to_xyxy(masks=masks)
        detections = sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)
        return detections

    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._prompted = False
        # The predictor state is cleared when load_first_frame is called again

    @property
    def is_initialized(self) -> bool:
        """Check if tracker has been prompted."""
        return self._prompted
