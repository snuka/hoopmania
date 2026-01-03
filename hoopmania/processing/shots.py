"""Shot event detection and tracking."""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np

from sports.basketball import ShotEventTracker as SportsShotEventTracker

from hoopmania.config import (
    SHOT_RESET_TIME_SECONDS,
    SHOT_MIN_FRAMES_BETWEEN_STARTS,
    SHOT_COOLDOWN_AFTER_MADE,
)


@dataclass
class ShotEvent:
    """Represents a detected shot event."""

    frame_idx: int
    shot_type: str  # "jump_shot" or "layup_dunk"
    made: bool
    court_position: Optional[np.ndarray] = None
    player_tracker_id: Optional[int] = None
    team_id: Optional[int] = None


class ShotDetector:
    """Detects and tracks shot events in basketball video.

    Uses detection of shot poses (jump shot, layup/dunk) combined
    with ball-in-basket detection to classify made/missed shots.
    """

    def __init__(
        self,
        fps: float,
        reset_time_seconds: float = SHOT_RESET_TIME_SECONDS,
        min_frames_between_starts: float = SHOT_MIN_FRAMES_BETWEEN_STARTS,
        cooldown_after_made: float = SHOT_COOLDOWN_AFTER_MADE,
    ):
        """Initialize shot detector.

        Args:
            fps: Video frames per second
            reset_time_seconds: Time to reset shot state if no basket
            min_frames_between_starts: Minimum time between shot starts
            cooldown_after_made: Cooldown time after made basket
        """
        self.fps = fps
        self._tracker = SportsShotEventTracker(
            reset_time_frames=int(fps * reset_time_seconds),
            minimum_frames_between_starts=int(fps * min_frames_between_starts),
            cooldown_frames_after_made=int(fps * cooldown_after_made),
        )
        self._events: List[ShotEvent] = []

    def update(
        self,
        frame_idx: int,
        has_jump_shot: bool,
        has_layup_dunk: bool,
        has_ball_in_basket: bool,
        court_position: Optional[np.ndarray] = None,
        player_tracker_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> Optional[List[dict]]:
        """Update shot tracker with current frame detections.

        Args:
            frame_idx: Current frame index
            has_jump_shot: Whether jump shot pose detected
            has_layup_dunk: Whether layup/dunk pose detected
            has_ball_in_basket: Whether ball in basket detected
            court_position: Shot location in court coordinates
            player_tracker_id: Tracker ID of shooter
            team_id: Team ID of shooter

        Returns:
            List of shot events if any completed, None otherwise
        """
        events = self._tracker.update(
            frame_index=frame_idx,
            has_jump_shot=has_jump_shot,
            has_layup_dunk=has_layup_dunk,
            has_ball_in_basket=has_ball_in_basket,
        )

        if events:
            shot_events = []
            for event in events:
                shot_event = ShotEvent(
                    frame_idx=frame_idx,
                    shot_type=event.get("type", "unknown"),
                    made=event.get("made", False),
                    court_position=court_position,
                    player_tracker_id=player_tracker_id,
                    team_id=team_id,
                )
                self._events.append(shot_event)
                shot_events.append(event)
            return shot_events

        return None

    @property
    def events(self) -> List[ShotEvent]:
        """Get all detected shot events."""
        return self._events.copy()

    @property
    def made_shots(self) -> List[ShotEvent]:
        """Get all made shots."""
        return [e for e in self._events if e.made]

    @property
    def missed_shots(self) -> List[ShotEvent]:
        """Get all missed shots."""
        return [e for e in self._events if not e.made]

    def get_shot_positions(self) -> dict:
        """Get shot positions grouped by result.

        Returns:
            Dict with 'made' and 'missed' keys containing
            arrays of court positions
        """
        made_positions = [
            e.court_position for e in self.made_shots
            if e.court_position is not None
        ]
        missed_positions = [
            e.court_position for e in self.missed_shots
            if e.court_position is not None
        ]

        return {
            "made": np.array(made_positions) if made_positions else np.array([]).reshape(0, 2),
            "missed": np.array(missed_positions) if missed_positions else np.array([]).reshape(0, 2),
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._events = []
        # Note: SportsShotEventTracker doesn't have a public reset method
        # Reinitialize if needed
