"""Path cleaning and smoothing utilities."""

from typing import Tuple
import numpy as np

from sports import clean_paths as sports_clean_paths

from hoopmania.config import PATH_SMOOTHING_WINDOW, PATH_SMOOTHING_POLY


class PathCleaner:
    """Cleans and smooths player movement paths.

    Handles:
    - Jump detection using robust speed analysis
    - Removal of short abnormal runs
    - Linear interpolation for missing segments
    - Savitzky-Golay smoothing for natural movement
    """

    def __init__(
        self,
        jump_sigma: float = 3.5,
        min_jump_dist: float = 0.6,
        max_jump_run: int = 18,
        pad_around_runs: int = 2,
        smooth_window: int = PATH_SMOOTHING_WINDOW,
        smooth_poly: int = PATH_SMOOTHING_POLY,
    ):
        """Initialize path cleaner.

        Args:
            jump_sigma: Sigma threshold for jump detection
            min_jump_dist: Minimum distance to consider as jump
            max_jump_run: Maximum length of abnormal run to remove
            pad_around_runs: Frames to pad around removed runs
            smooth_window: Savitzky-Golay window size
            smooth_poly: Savitzky-Golay polynomial order
        """
        self.jump_sigma = jump_sigma
        self.min_jump_dist = min_jump_dist
        self.max_jump_run = max_jump_run
        self.pad_around_runs = pad_around_runs
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly

    def clean(
        self,
        video_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clean player movement paths.

        Args:
            video_xy: Position array of shape (frames, players, 2)

        Returns:
            Tuple of (cleaned_xy, edited_mask) where edited_mask
            indicates which frames were modified
        """
        return sports_clean_paths(
            video_xy,
            jump_sigma=self.jump_sigma,
            min_jump_dist=self.min_jump_dist,
            max_jump_run=self.max_jump_run,
            pad_around_runs=self.pad_around_runs,
            smooth_window=self.smooth_window,
            smooth_poly=self.smooth_poly,
        )

    @staticmethod
    def split_true_runs(
        mask: np.ndarray,
        coords: np.ndarray,
        player_idx: int = 0,
    ) -> list:
        """Split coordinates into continuous runs based on mask.

        Useful for visualizing which path segments were edited.

        Args:
            mask: Boolean mask of edited frames (frames, players)
            coords: Position array (frames, players, 2)
            player_idx: Index of player to extract

        Returns:
            List of coordinate arrays for each continuous run
        """
        mask = mask[:, player_idx] if mask.ndim > 1 else mask.squeeze()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        return [coords[g, player_idx, :] for g in groups]
