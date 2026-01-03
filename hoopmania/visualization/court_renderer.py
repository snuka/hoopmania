"""Court visualization and rendering utilities."""

from typing import Dict, List, Optional
import numpy as np
import supervision as sv

from sports import MeasurementUnit
from sports.basketball import (
    CourtConfiguration,
    League,
    draw_court,
    draw_points_on_court,
    draw_paths_on_court,
    draw_made_and_miss_on_court,
)


class CourtRenderer:
    """Renders 2D basketball court visualizations.

    Creates bird's-eye view court diagrams with:
    - Player positions as colored dots
    - Movement paths as trails
    - Shot charts with made/missed indicators
    """

    def __init__(
        self,
        league: League = League.NBA,
        measurement_unit: MeasurementUnit = MeasurementUnit.FEET,
    ):
        """Initialize court renderer.

        Args:
            league: Basketball league for court configuration
            measurement_unit: Unit for measurements
        """
        self.league = league
        self.measurement_unit = measurement_unit
        self._config = None
        self._base_court = None

    @property
    def config(self) -> CourtConfiguration:
        """Get court configuration."""
        if self._config is None:
            self._config = CourtConfiguration(
                league=self.league,
                measurement_unit=self.measurement_unit,
            )
        return self._config

    def get_base_court(self) -> np.ndarray:
        """Get blank court image.

        Returns:
            BGR court image
        """
        return draw_court(config=self.config)

    @property
    def court_size(self) -> tuple:
        """Get court image dimensions (height, width)."""
        court = self.get_base_court()
        return court.shape[:2]

    def draw_players(
        self,
        positions: np.ndarray,
        team_colors: Dict[int, str],
        team_ids: np.ndarray,
        court: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw player positions on court.

        Args:
            positions: Player positions (N, 2)
            team_colors: Mapping of team ID to hex color
            team_ids: Team ID for each player
            court: Base court image (creates new if None)

        Returns:
            Court image with player dots
        """
        if court is None:
            court = self.get_base_court()

        for team_id in np.unique(team_ids):
            mask = team_ids == team_id
            team_positions = positions[mask]
            color = sv.Color.from_hex(team_colors.get(int(team_id), "#FFFFFF"))

            court = draw_points_on_court(
                config=self.config,
                xy=team_positions,
                fill_color=color,
                court=court,
            )

        return court

    def draw_paths(
        self,
        paths: List[np.ndarray],
        color: Optional[sv.Color] = None,
        court: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw movement paths on court.

        Args:
            paths: List of path arrays, each (frames, 2)
            color: Path color (default green)
            court: Base court image

        Returns:
            Court image with paths
        """
        if court is None:
            court = self.get_base_court()

        color = color or sv.Color.GREEN

        return draw_paths_on_court(
            config=self.config,
            paths=paths,
            color=color,
            court=court,
        )

    def draw_shot_chart(
        self,
        made_positions: np.ndarray,
        missed_positions: Optional[np.ndarray] = None,
        made_color: str = "#007A33",
        missed_color: str = "#CE1141",
        court: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw shot chart with made/missed indicators.

        Args:
            made_positions: Made shot positions (N, 2)
            missed_positions: Missed shot positions (M, 2)
            made_color: Color for made shots
            missed_color: Color for missed shots
            court: Base court image

        Returns:
            Court image with shot chart
        """
        if court is None:
            court = self.get_base_court()

        if missed_positions is None:
            missed_positions = np.array([]).reshape(0, 2)

        return draw_made_and_miss_on_court(
            config=self.config,
            made_xy=made_positions,
            miss_xy=missed_positions,
            made_color=sv.Color.from_hex(made_color),
            miss_color=sv.Color.from_hex(missed_color),
            made_size=25,
            miss_size=25,
            made_thickness=6,
            miss_thickness=6,
            line_thickness=4,
            court=court,
        )

    def create_frame_for_positions(
        self,
        positions: np.ndarray,
        team_colors: Dict[int, str],
        team_ids: np.ndarray,
    ) -> np.ndarray:
        """Create single court frame showing player positions.

        Args:
            positions: Player positions for this frame (N, 2)
            team_colors: Team ID to color mapping
            team_ids: Team ID for each player

        Returns:
            Court image with player positions
        """
        return self.draw_players(
            positions=positions,
            team_colors=team_colors,
            team_ids=team_ids,
        )
