"""Court mapping and homography utilities."""

from typing import Optional, Tuple
import numpy as np
import supervision as sv

from sports import ViewTransformer, MeasurementUnit
from sports.basketball import CourtConfiguration, League


class CourtMapper:
    """Maps player positions from video frame to 2D court coordinates.

    Uses homography estimation based on detected court keypoints.
    """

    def __init__(
        self,
        league: League = League.NBA,
        measurement_unit: MeasurementUnit = MeasurementUnit.FEET,
    ):
        """Initialize court mapper.

        Args:
            league: Basketball league for court configuration
            measurement_unit: Unit for court measurements
        """
        self.league = league
        self.measurement_unit = measurement_unit
        self._config = None
        self._transformer = None

    @property
    def config(self) -> CourtConfiguration:
        """Get court configuration."""
        if self._config is None:
            self._config = CourtConfiguration(
                league=self.league,
                measurement_unit=self.measurement_unit,
            )
        return self._config

    @property
    def court_vertices(self) -> np.ndarray:
        """Get court vertex coordinates."""
        return np.array(self.config.vertices)

    def update_homography(
        self,
        frame_landmarks: np.ndarray,
        landmark_mask: np.ndarray,
    ) -> bool:
        """Update homography matrix from detected landmarks.

        Args:
            frame_landmarks: Detected keypoint coordinates in frame
            landmark_mask: Boolean mask indicating which court vertices
                          correspond to detected landmarks

        Returns:
            True if homography was successfully computed
        """
        if len(frame_landmarks) < 4:
            return False

        court_landmarks = self.court_vertices[landmark_mask]

        try:
            self._transformer = ViewTransformer(
                source=frame_landmarks,
                target=court_landmarks,
            )
            return True
        except Exception:
            return False

    def transform_points(
        self,
        frame_points: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Transform points from frame coordinates to court coordinates.

        Args:
            frame_points: Points in frame coordinates (N, 2)

        Returns:
            Points in court coordinates (N, 2) or None if no homography
        """
        if self._transformer is None:
            return None

        if len(frame_points) == 0:
            return np.array([]).reshape(0, 2)

        return self._transformer.transform_points(points=frame_points)

    def get_player_court_positions(
        self,
        detections: sv.Detections,
        anchor: sv.Position = sv.Position.BOTTOM_CENTER,
    ) -> Optional[np.ndarray]:
        """Get player positions in court coordinates.

        Args:
            detections: Player detections
            anchor: Anchor point to use for position

        Returns:
            Court positions (N, 2) or None if no homography
        """
        frame_xy = detections.get_anchors_coordinates(anchor=anchor)
        return self.transform_points(frame_xy)

    @property
    def has_homography(self) -> bool:
        """Check if homography matrix is available."""
        return self._transformer is not None

    def reset(self) -> None:
        """Reset homography state."""
        self._transformer = None
