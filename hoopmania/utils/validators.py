"""Validation utilities for jersey numbers and detection matching."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv

from sports import ConsecutiveValueTracker as SportsConsecutiveValueTracker

from hoopmania.config import CONSECUTIVE_VALIDATION_THRESHOLD


class ConsecutiveValueTracker:
    """Track values across frames and validate with consecutive agreement.

    Used for:
    - Jersey number validation (requires N consecutive reads)
    - Team assignment validation
    """

    def __init__(self, n_consecutive: int = CONSECUTIVE_VALIDATION_THRESHOLD):
        """Initialize tracker.

        Args:
            n_consecutive: Number of consecutive matches to confirm value
        """
        self._tracker = SportsConsecutiveValueTracker(n_consecutive=n_consecutive)

    def update(
        self,
        tracker_ids: List[int],
        values: List,
    ) -> None:
        """Update tracker with new observations.

        Args:
            tracker_ids: List of object tracker IDs
            values: Corresponding values for each tracker
        """
        self._tracker.update(tracker_ids=tracker_ids, values=values)

    def get_validated(
        self,
        tracker_ids: np.ndarray,
    ) -> List[Optional[str]]:
        """Get validated values for tracker IDs.

        Args:
            tracker_ids: Array of tracker IDs to query

        Returns:
            List of validated values (None if not yet validated)
        """
        return self._tracker.get_validated(tracker_ids=tracker_ids)

    def reset(self) -> None:
        """Reset all tracked values."""
        self._tracker = SportsConsecutiveValueTracker(
            n_consecutive=self._tracker.n_consecutive
        )


def coords_above_threshold(
    matrix: np.ndarray,
    threshold: float,
    sort_desc: bool = True,
) -> List[Tuple[int, int]]:
    """Return all (row_index, col_index) where value > threshold.

    Args:
        matrix: 2D array of values
        threshold: Minimum value threshold
        sort_desc: Whether to sort by value descending

    Returns:
        List of (row, col) tuples
    """
    A = np.asarray(matrix)
    rows, cols = np.where(A > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc:
        pairs.sort(key=lambda rc: A[rc[0], rc[1]], reverse=True)
    return pairs


def match_numbers_to_players(
    player_detections: sv.Detections,
    number_detections: sv.Detections,
    frame_shape: Tuple[int, int],
    ios_threshold: float = 0.9,
) -> Tuple[List[int], List[int]]:
    """Match number detections to player detections using IoS.

    Uses Intersection over Smaller Area to match numbers that lie
    within player masks.

    Args:
        player_detections: Player detections with masks
        number_detections: Number box detections
        frame_shape: (height, width) of frame
        ios_threshold: Minimum IoS for match

    Returns:
        Tuple of (player_indices, number_indices) for matched pairs
    """
    if len(player_detections) == 0 or len(number_detections) == 0:
        return [], []

    frame_h, frame_w = frame_shape

    # Convert number boxes to masks
    number_masks = sv.xyxy_to_mask(
        boxes=number_detections.xyxy,
        resolution_wh=(frame_w, frame_h)
    )
    number_detections_with_masks = sv.Detections(
        xyxy=number_detections.xyxy,
        mask=number_masks,
    )

    # Compute IoS matrix
    ios_matrix = sv.mask_iou_batch(
        masks_true=player_detections.mask,
        masks_detection=number_detections_with_masks.mask,
        overlap_metric=sv.OverlapMetric.IOS,
    )

    # Get pairs above threshold
    pairs = coords_above_threshold(ios_matrix, ios_threshold)

    if not pairs:
        return [], []

    player_idx, number_idx = zip(*pairs)
    return list(player_idx), list(number_idx)


def get_player_names(
    numbers: np.ndarray,
    teams: np.ndarray,
    team_names: Dict[int, str],
    team_rosters: Dict[str, Dict[str, str]],
) -> List[str]:
    """Get player names from jersey numbers and team assignments.

    Args:
        numbers: Array of jersey number strings
        teams: Array of team IDs
        team_names: Mapping of team ID to team name
        team_rosters: Nested dict of team name -> number -> player name

    Returns:
        List of formatted labels ("#{number} {name}")
    """
    labels = []
    for number, team in zip(numbers, teams):
        team_name = team_names.get(int(team), f"Team {team}")
        roster = team_rosters.get(team_name, {})
        player_name = roster.get(str(number), "")
        labels.append(f"#{number} {player_name}")
    return labels
