"""Custom annotators for basketball video visualization."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import supervision as sv

from hoopmania.config import ANNOTATION_COLORS, FONTS_DIR


class TeamAnnotator:
    """Team-colored annotators for player visualization.

    Provides consistent team-colored masks, boxes, and labels.
    """

    def __init__(
        self,
        team_colors: Dict[int, str],
        font_path: Optional[Path] = None,
        font_size: int = 40,
    ):
        """Initialize team annotator.

        Args:
            team_colors: Mapping of team ID to hex color
            font_path: Path to TTF font for labels
            font_size: Font size for labels
        """
        self.team_colors = team_colors
        self.font_path = font_path or FONTS_DIR / "Staatliches-Regular.ttf"
        self.font_size = font_size

        # Create color palette
        colors = [team_colors.get(i, "#FFFFFF") for i in range(len(team_colors))]
        self._palette = sv.ColorPalette.from_hex(colors)

        # Initialize annotators
        self._mask_annotator = sv.MaskAnnotator(
            color=self._palette,
            opacity=0.5,
            color_lookup=sv.ColorLookup.INDEX,
        )
        self._box_annotator = sv.BoxAnnotator(
            color=self._palette,
            thickness=2,
            color_lookup=sv.ColorLookup.INDEX,
        )
        self._label_annotator = None
        self._init_label_annotator()

    def _init_label_annotator(self):
        """Initialize label annotator with font."""
        if self.font_path.exists():
            self._label_annotator = sv.RichLabelAnnotator(
                font_path=str(self.font_path),
                font_size=self.font_size,
                color=self._palette,
                text_color=sv.Color.WHITE,
                text_position=sv.Position.BOTTOM_CENTER,
                text_offset=(0, 10),
                color_lookup=sv.ColorLookup.INDEX,
            )
        else:
            # Fallback to basic label annotator
            self._label_annotator = sv.LabelAnnotator(
                color=self._palette,
                text_color=sv.Color.WHITE,
                text_position=sv.Position.BOTTOM_CENTER,
                color_lookup=sv.ColorLookup.INDEX,
            )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        team_ids: np.ndarray,
        labels: Optional[List[str]] = None,
        show_masks: bool = True,
        show_boxes: bool = False,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Annotate frame with team-colored visualizations.

        Args:
            frame: BGR image
            detections: Player detections
            team_ids: Team ID for each detection
            labels: Optional labels for each detection
            show_masks: Whether to draw masks
            show_boxes: Whether to draw boxes
            show_labels: Whether to draw labels

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        if show_masks and detections.mask is not None:
            annotated = self._mask_annotator.annotate(
                scene=annotated,
                detections=detections,
                custom_color_lookup=team_ids,
            )

        if show_boxes:
            annotated = self._box_annotator.annotate(
                scene=annotated,
                detections=detections,
                custom_color_lookup=team_ids,
            )

        if show_labels and labels is not None:
            annotated = self._label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels,
                custom_color_lookup=team_ids,
            )

        return annotated


class DefaultAnnotator:
    """Default annotators using track ID for colors."""

    def __init__(self, colors: Optional[List[str]] = None):
        """Initialize default annotator.

        Args:
            colors: List of hex colors (uses default if None)
        """
        colors = colors or ANNOTATION_COLORS
        self._palette = sv.ColorPalette.from_hex(colors)

        self._mask_annotator = sv.MaskAnnotator(
            color=self._palette,
            color_lookup=sv.ColorLookup.TRACK,
            opacity=0.5,
        )
        self._box_annotator = sv.BoxAnnotator(
            color=self._palette,
            color_lookup=sv.ColorLookup.TRACK,
            thickness=2,
        )
        self._label_annotator = sv.LabelAnnotator(
            color=self._palette,
            color_lookup=sv.ColorLookup.TRACK,
            text_color=sv.Color.BLACK,
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
        show_masks: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Annotate frame with default colors.

        Args:
            frame: BGR image
            detections: Detections to visualize
            labels: Optional labels
            show_masks: Whether to draw masks
            show_boxes: Whether to draw boxes
            show_labels: Whether to draw labels

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        if show_masks and detections.mask is not None:
            annotated = self._mask_annotator.annotate(
                scene=annotated,
                detections=detections,
            )

        if show_boxes:
            annotated = self._box_annotator.annotate(
                scene=annotated,
                detections=detections,
            )

        if show_labels and labels is not None:
            annotated = self._label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels,
            )

        return annotated
