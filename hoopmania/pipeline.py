"""Main basketball analysis pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
import supervision as sv
from tqdm import tqdm

from hoopmania.config import (
    TEAM_COLORS,
    TEAM_ROSTERS,
    PLAYER_CLASS_IDS,
    OCR_FRAME_INTERVAL,
    OUTPUTS_DIR,
)
from hoopmania.models import (
    PlayerDetector,
    SAM2Tracker,
    TeamClassifier,
    JerseyOCR,
    CourtKeypointDetector,
)
from hoopmania.processing import (
    VideoProcessor,
    CourtMapper,
    PathCleaner,
    ShotDetector,
)
from hoopmania.visualization import TeamAnnotator, CourtRenderer
from hoopmania.utils.validators import (
    ConsecutiveValueTracker,
    match_numbers_to_players,
    get_player_names,
)


@dataclass
class AnalysisConfig:
    """Configuration for basketball analysis."""

    team1_name: str = "Team A"
    team2_name: str = "Team B"
    team1_color: str = "#007A33"
    team2_color: str = "#006BB6"
    skip_ocr: bool = False
    skip_court_map: bool = False
    skip_shot_detection: bool = False
    ocr_interval: int = OCR_FRAME_INTERVAL
    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR)


@dataclass
class AnalysisResult:
    """Results from basketball analysis."""

    video_path: Path
    annotated_video_path: Optional[Path] = None
    court_map_video_path: Optional[Path] = None
    shot_chart_path: Optional[Path] = None
    tracking_data_path: Optional[Path] = None
    player_data: Dict = field(default_factory=dict)
    shot_events: List = field(default_factory=list)


class BasketballAnalyzer:
    """Main basketball video analysis pipeline.

    Orchestrates:
    - Player detection and tracking
    - Team classification
    - Jersey number recognition
    - Court position mapping
    - Shot event detection
    - Visualization generation
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()

        # Models (lazy loaded)
        self._detector = None
        self._tracker = None
        self._team_classifier = None
        self._jersey_ocr = None
        self._keypoint_detector = None

        # Processing
        self._court_mapper = None
        self._path_cleaner = None
        self._shot_detector = None

        # Visualization
        self._team_annotator = None
        self._court_renderer = None

        # State
        self._team_names = {}
        self._team_colors = {}

    def _init_models(self):
        """Initialize all models."""
        print("Loading models...")
        self._detector = PlayerDetector()
        self._tracker = SAM2Tracker()
        self._team_classifier = TeamClassifier()

        if not self.config.skip_ocr:
            self._jersey_ocr = JerseyOCR()

        if not self.config.skip_court_map:
            self._keypoint_detector = CourtKeypointDetector()
            self._court_mapper = CourtMapper()
            self._court_renderer = CourtRenderer()

        self._path_cleaner = PathCleaner()

    def _setup_team_mapping(self, team_0_name: str, team_1_name: str):
        """Set up team name and color mappings."""
        self._team_names = {0: team_0_name, 1: team_1_name}
        self._team_colors = {
            0: TEAM_COLORS.get(team_0_name, self.config.team1_color),
            1: TEAM_COLORS.get(team_1_name, self.config.team2_color),
        }
        self._team_classifier.set_team_names(team_0_name, team_1_name)

        # Initialize team annotator
        self._team_annotator = TeamAnnotator(team_colors=self._team_colors)

    def _collect_training_samples(
        self,
        video_processor: VideoProcessor,
        stride: int = 30,
    ) -> List[np.ndarray]:
        """Collect player crops for team classifier training."""
        crops = []

        print("Collecting training samples for team classification...")
        for frame in tqdm(video_processor.get_frames(stride=stride)):
            detections = self._detector.detect_players(frame)
            if len(detections) > 0:
                boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
                for box in boxes:
                    crops.append(sv.crop_image(frame, box))

        return crops

    def _train_team_classifier(self, crops: List[np.ndarray]) -> np.ndarray:
        """Train team classifier and get initial predictions."""
        print("Training team classifier...")
        self._team_classifier.fit(crops)
        return self._team_classifier.predict(crops)

    def analyze(
        self,
        video_path: Path,
        team1_name: Optional[str] = None,
        team2_name: Optional[str] = None,
    ) -> AnalysisResult:
        """Run full analysis on a basketball video.

        Args:
            video_path: Path to input video
            team1_name: Name of first team
            team2_name: Name of second team

        Returns:
            AnalysisResult with paths to generated outputs
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Set up team names
        team1 = team1_name or self.config.team1_name
        team2 = team2_name or self.config.team2_name

        # Create output directory
        output_dir = self.config.output_dir / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        result = AnalysisResult(video_path=video_path)

        # Initialize models
        self._init_models()

        # Create video processor
        video_processor = VideoProcessor(video_path)
        fps = video_processor.fps

        # Initialize shot detector
        if not self.config.skip_shot_detection:
            self._shot_detector = ShotDetector(fps=fps)

        # Phase 1: Collect training samples and train team classifier
        crops = self._collect_training_samples(video_processor)
        if len(crops) > 0:
            teams = self._train_team_classifier(crops)

            # NOTE: User may need to verify team mapping after seeing initial results
            # For now, we use the configured order
            self._setup_team_mapping(team1, team2)
        else:
            print("Warning: No player crops collected for team classification")
            self._setup_team_mapping(team1, team2)

        # Phase 2: Full video analysis
        print("Running full video analysis...")
        (
            frames_history,
            detections_history,
            teams_history,
            court_positions_history,
        ) = self._run_tracking_pass(video_processor)

        # Phase 3: Generate outputs
        print("Generating outputs...")

        # Generate annotated video
        annotated_path = self._generate_annotated_video(
            video_processor,
            frames_history,
            detections_history,
            teams_history,
            output_dir,
        )
        result.annotated_video_path = annotated_path

        # Generate court map video
        if not self.config.skip_court_map and court_positions_history:
            court_map_path = self._generate_court_map_video(
                video_processor,
                court_positions_history,
                teams_history,
                output_dir,
            )
            result.court_map_video_path = court_map_path

        # Generate shot chart
        if not self.config.skip_shot_detection and self._shot_detector:
            shot_chart_path = self._generate_shot_chart(output_dir)
            result.shot_chart_path = shot_chart_path
            result.shot_events = [
                {
                    "frame": e.frame_idx,
                    "type": e.shot_type,
                    "made": e.made,
                    "position": e.court_position.tolist() if e.court_position is not None else None,
                }
                for e in self._shot_detector.events
            ]

        # Export tracking data
        tracking_data_path = self._export_tracking_data(
            video_processor,
            detections_history,
            teams_history,
            court_positions_history,
            output_dir,
        )
        result.tracking_data_path = tracking_data_path

        print(f"Analysis complete! Outputs saved to: {output_dir}")
        return result

    def _run_tracking_pass(
        self,
        video_processor: VideoProcessor,
    ) -> Tuple[List, List, np.ndarray, List]:
        """Run full tracking pass over video.

        Returns:
            Tuple of (frames, detections, teams, court_positions)
        """
        frames_history = []
        detections_history = []
        court_positions_history = []

        # Validators
        number_validator = ConsecutiveValueTracker(n_consecutive=3)
        team_validator = ConsecutiveValueTracker(n_consecutive=1)

        # Get first frame and initialize
        frame_generator = video_processor.get_frames()
        first_frame = next(frame_generator)

        # Initial detection
        detections = self._detector.detect_players(first_frame)
        detections.tracker_id = np.arange(1, len(detections) + 1)

        # Get team assignments for first frame
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
        crops = [sv.crop_image(first_frame, box) for box in boxes]
        teams = self._team_classifier.predict(crops)
        team_validator.update(tracker_ids=detections.tracker_id.tolist(), values=teams.tolist())

        # Initialize tracker
        self._tracker.prompt_first_frame(first_frame, detections)

        # Process all frames
        frame_generator = video_processor.get_frames()
        for frame_idx, frame in tqdm(enumerate(frame_generator), total=video_processor.total_frames):
            frames_history.append(frame)

            # Track
            player_detections = self._tracker.propagate(frame)
            detections_history.append(player_detections)

            # OCR (at intervals)
            if not self.config.skip_ocr and frame_idx % self.config.ocr_interval == 0:
                self._process_jersey_numbers(
                    frame, player_detections, number_validator
                )

            # Court mapping
            if not self.config.skip_court_map:
                court_xy = self._process_court_positions(frame, player_detections)
                court_positions_history.append(court_xy)

            # Shot detection
            if not self.config.skip_shot_detection:
                shot_events = self._detector.detect_shot_events(frame)
                self._shot_detector.update(
                    frame_idx=frame_idx,
                    **shot_events,
                )

        # Get validated teams for all tracked players
        all_tracker_ids = set()
        for det in detections_history:
            if det.tracker_id is not None:
                all_tracker_ids.update(det.tracker_id.tolist())

        teams_array = team_validator.get_validated(np.array(list(all_tracker_ids)))

        return frames_history, detections_history, teams, court_positions_history

    def _process_jersey_numbers(
        self,
        frame: np.ndarray,
        player_detections: sv.Detections,
        number_validator: ConsecutiveValueTracker,
    ):
        """Process jersey numbers for a frame."""
        frame_h, frame_w = frame.shape[:2]

        # Detect number regions
        number_detections = self._detector.detect_numbers(frame)
        if len(number_detections) == 0:
            return

        # Prepare crops and recognize
        crops = self._jersey_ocr.prepare_crops(frame, number_detections)
        numbers = self._jersey_ocr.recognize_batch(crops)

        # Match to players
        number_detections.mask = sv.xyxy_to_mask(
            boxes=number_detections.xyxy,
            resolution_wh=(frame_w, frame_h)
        )

        player_idx, number_idx = match_numbers_to_players(
            player_detections,
            number_detections,
            (frame_h, frame_w),
        )

        if player_idx:
            matched_tracker_ids = [player_detections.tracker_id[i] for i in player_idx]
            matched_numbers = [numbers[i] for i in number_idx]
            number_validator.update(
                tracker_ids=matched_tracker_ids,
                values=matched_numbers,
            )

    def _process_court_positions(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
    ) -> Optional[np.ndarray]:
        """Process court positions for a frame."""
        frame_landmarks, mask = self._keypoint_detector.get_landmarks_for_homography(frame)

        if frame_landmarks is not None:
            if self._court_mapper.update_homography(frame_landmarks, mask):
                return self._court_mapper.get_player_court_positions(detections)

        return None

    def _generate_annotated_video(
        self,
        video_processor: VideoProcessor,
        frames_history: List[np.ndarray],
        detections_history: List[sv.Detections],
        teams: np.ndarray,
        output_dir: Path,
    ) -> Path:
        """Generate annotated video with team colors and labels."""
        output_path = output_dir / f"{video_processor.video_path.stem}_annotated.mp4"

        number_validator = ConsecutiveValueTracker(n_consecutive=3)
        team_validator = ConsecutiveValueTracker(n_consecutive=1)

        # Initialize team validator with first frame data
        if len(detections_history) > 0:
            first_det = detections_history[0]
            if first_det.tracker_id is not None:
                team_validator.update(
                    tracker_ids=first_det.tracker_id.tolist(),
                    values=[teams[i % len(teams)] for i in range(len(first_det))],
                )

        with sv.VideoSink(str(output_path), video_processor.info) as sink:
            for frame, detections in tqdm(
                zip(frames_history, detections_history),
                total=len(frames_history),
                desc="Generating annotated video",
            ):
                # Filter small detections
                if detections.area is not None:
                    detections = detections[detections.area > 100]

                if len(detections) == 0:
                    sink.write_frame(frame)
                    continue

                # Get team assignments
                team_ids = team_validator.get_validated(detections.tracker_id)
                team_ids = np.array([t if t is not None else 0 for t in team_ids]).astype(int)

                # Get number labels
                numbers = number_validator.get_validated(detections.tracker_id)
                labels = get_player_names(
                    numbers=np.array([n if n else "?" for n in numbers]),
                    teams=team_ids,
                    team_names=self._team_names,
                    team_rosters=TEAM_ROSTERS,
                )

                # Annotate
                annotated = self._team_annotator.annotate(
                    frame=frame,
                    detections=detections,
                    team_ids=team_ids,
                    labels=labels,
                )

                sink.write_frame(annotated)

        # Compress
        compressed_path = VideoProcessor.compress(output_path)
        return compressed_path

    def _generate_court_map_video(
        self,
        video_processor: VideoProcessor,
        court_positions_history: List[Optional[np.ndarray]],
        teams: np.ndarray,
        output_dir: Path,
    ) -> Path:
        """Generate 2D court map video."""
        output_path = output_dir / f"{video_processor.video_path.stem}_court_map.mp4"

        court_h, court_w = self._court_renderer.court_size

        # Create video info for court size
        court_video_info = sv.VideoInfo(
            width=court_w,
            height=court_h,
            fps=video_processor.fps,
        )

        with sv.VideoSink(str(output_path), court_video_info) as sink:
            for positions in tqdm(
                court_positions_history,
                desc="Generating court map video",
            ):
                if positions is None:
                    court = self._court_renderer.get_base_court()
                else:
                    # Assign teams based on position in array
                    team_ids = np.array([
                        teams[i % len(teams)]
                        for i in range(len(positions))
                    ])

                    court = self._court_renderer.draw_players(
                        positions=positions,
                        team_colors=self._team_colors,
                        team_ids=team_ids,
                    )

                sink.write_frame(court)

        compressed_path = VideoProcessor.compress(output_path)
        return compressed_path

    def _generate_shot_chart(self, output_dir: Path) -> Path:
        """Generate shot chart image."""
        output_path = output_dir / "shot_chart.png"

        positions = self._shot_detector.get_shot_positions()

        chart = self._court_renderer.draw_shot_chart(
            made_positions=positions["made"],
            missed_positions=positions["missed"],
        )

        import cv2
        cv2.imwrite(str(output_path), chart)

        return output_path

    def _export_tracking_data(
        self,
        video_processor: VideoProcessor,
        detections_history: List[sv.Detections],
        teams: np.ndarray,
        court_positions_history: List[Optional[np.ndarray]],
        output_dir: Path,
    ) -> Path:
        """Export tracking data to JSON."""
        output_path = output_dir / "tracking_data.json"

        data = {
            "video_info": {
                "path": str(video_processor.video_path),
                "fps": video_processor.fps,
                "width": video_processor.width,
                "height": video_processor.height,
                "total_frames": video_processor.total_frames,
            },
            "team_names": self._team_names,
            "frames": [],
        }

        for frame_idx, detections in enumerate(detections_history):
            frame_data = {
                "frame_idx": frame_idx,
                "players": [],
            }

            if detections.tracker_id is not None:
                for i, tracker_id in enumerate(detections.tracker_id):
                    player_data = {
                        "tracker_id": int(tracker_id),
                        "bbox": detections.xyxy[i].tolist(),
                    }

                    if frame_idx < len(court_positions_history):
                        positions = court_positions_history[frame_idx]
                        if positions is not None and i < len(positions):
                            player_data["court_position"] = positions[i].tolist()

                    frame_data["players"].append(player_data)

            data["frames"].append(frame_data)

        if self._shot_detector:
            data["shot_events"] = [
                {
                    "frame": e.frame_idx,
                    "type": e.shot_type,
                    "made": e.made,
                    "position": e.court_position.tolist() if e.court_position is not None else None,
                }
                for e in self._shot_detector.events
            ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path
