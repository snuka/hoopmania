"""Team classification using SigLIP embeddings and K-means clustering."""

from typing import List, Optional
import numpy as np
import supervision as sv

# Import from roboflow sports library
from sports import TeamClassifier as SportsTeamClassifier


class TeamClassifier:
    """Team classifier using SigLIP embeddings and K-means clustering.

    Uses the sports library's TeamClassifier which:
    - Extracts embeddings using SigLIP
    - Reduces dimensions with UMAP
    - Clusters with K-means (k=2)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the team classifier.

        Args:
            device: Device to run inference on
        """
        self.device = device
        self._classifier = SportsTeamClassifier(device=device)
        self._fitted = False
        self._team_names = {}

    def fit(self, crops: List[np.ndarray]) -> None:
        """Train the classifier on player crops.

        Args:
            crops: List of player crop images (central region recommended)
        """
        self._classifier.fit(crops)
        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team IDs for player crops.

        Args:
            crops: List of player crop images

        Returns:
            Array of team IDs (0 or 1)

        Raises:
            RuntimeError: If classifier not fitted
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        return np.array(self._classifier.predict(crops))

    def set_team_names(self, team_0: str, team_1: str) -> None:
        """Map team IDs to team names.

        Since clustering doesn't know which team is which,
        this must be set after inspecting initial predictions.

        Args:
            team_0: Name for team ID 0
            team_1: Name for team ID 1
        """
        self._team_names = {0: team_0, 1: team_1}

    def get_team_name(self, team_id: int) -> str:
        """Get team name for a team ID.

        Args:
            team_id: Team ID (0 or 1)

        Returns:
            Team name or "Team {id}" if not set
        """
        return self._team_names.get(team_id, f"Team {team_id}")

    @property
    def team_names(self) -> dict:
        """Get team name mapping."""
        return self._team_names.copy()

    @property
    def is_fitted(self) -> bool:
        """Check if classifier is fitted."""
        return self._fitted

    @staticmethod
    def extract_crops(
        frame: np.ndarray,
        detections: sv.Detections,
        scale_factor: float = 0.4,
    ) -> List[np.ndarray]:
        """Extract central crops from player detections.

        Args:
            frame: BGR image as numpy array
            detections: Player detections
            scale_factor: Scale factor for crop box (smaller = more central)

        Returns:
            List of crop images
        """
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=scale_factor)
        return [sv.crop_image(frame, box) for box in boxes]
