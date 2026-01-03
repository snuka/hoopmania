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
from hoopmania.utils.progress import ProgressTracker


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

    # === Manual Input Features for Improved Detection ===

    # 1. Jersey Number Whitelist - Valid jersey numbers per team
    #    OCR results will be filtered/validated against these sets
    #    Example: team1_jersey_numbers={"0", "7", "8", "11", "23"}
    team1_jersey_numbers: Optional[set] = None
    team2_jersey_numbers: Optional[set] = None

    # 2. Reference Frame Selection - Frame index for tracker initialization
    #    Use a frame where all players are clearly visible and spread out
    #    Default 0 means use first frame
    reference_frame: int = 0

    # 3. Team Assignment Seeds - Bounding boxes identifying team members
    #    Format: {team_name: [(x1, y1, x2, y2), ...]}
    #    These boxes should be from the reference frame
    #    Just 1-2 players per team is enough to anchor the classification
    #    Example: {"Boston Celtics": [(100, 200, 150, 400)], "New York Knicks": [(500, 200, 550, 400)]}
    team_seeds: Optional[Dict[str, List[Tuple[float, float, float, float]]]] = None


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

        # Progress tracking
        self._progress_tracker: Optional[ProgressTracker] = None

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

        total_samples = video_processor.total_frames // stride
        if self._progress_tracker:
            self._progress_tracker.start_phase("Collecting samples", total=total_samples)

        print("Collecting training samples for team classification...")
        sample_idx = 0
        for frame in tqdm(video_processor.get_frames(stride=stride), total=total_samples, desc="Collecting samples"):
            detections = self._detector.detect_players(frame)
            if len(detections) > 0:
                boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
                for box in boxes:
                    crops.append(sv.crop_image(frame, box))

            sample_idx += 1
            if self._progress_tracker:
                self._progress_tracker.set_step(sample_idx)

        return crops

    def _train_team_classifier(self, crops: List[np.ndarray]) -> np.ndarray:
        """Train team classifier and get initial predictions."""
        print("Training team classifier...")
        self._team_classifier.fit(crops)
        return self._team_classifier.predict(crops)

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union between two boxes.

        Args:
            box1: First box as (x1, y1, x2, y2)
            box2: Second box as (x1, y1, x2, y2)

        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _determine_team_mapping_from_seeds(
        self,
        video_processor: VideoProcessor,
        team1_name: str,
        team2_name: str,
    ) -> Tuple[str, str]:
        """Determine correct team mapping using seed annotations.

        Uses the team_seeds config to match annotated bounding boxes to
        detected players, then uses their cluster assignments to determine
        which cluster corresponds to which team.

        Args:
            video_processor: Video processor for reading frames
            team1_name: Name of first team
            team2_name: Name of second team

        Returns:
            Tuple of (team_for_cluster_0, team_for_cluster_1)
        """
        if not self.config.team_seeds:
            # No seeds provided, use default order
            return team1_name, team2_name

        print("Determining team mapping from seed annotations...")

        # Get reference frame
        reference_frame_idx = self.config.reference_frame
        if reference_frame_idx >= video_processor.total_frames:
            reference_frame_idx = 0

        ref_frame_generator = video_processor.get_frames(start=reference_frame_idx)
        reference_frame = next(ref_frame_generator)

        # Detect players on reference frame
        detections = self._detector.detect_players(reference_frame)
        if len(detections) == 0:
            print("  Warning: No players detected on reference frame, using default mapping")
            return team1_name, team2_name

        # Get team predictions for detected players
        boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
        crops = [sv.crop_image(reference_frame, box) for box in boxes]
        predictions = self._team_classifier.predict(crops)

        # Match seed boxes to detected players and collect cluster votes
        cluster_votes = {0: [], 1: []}  # cluster_id -> list of team names

        for team_name, seed_boxes in self.config.team_seeds.items():
            for seed_box in seed_boxes:
                seed_box_arr = np.array(seed_box)

                # Find best matching detection using IoU
                best_iou = 0
                best_idx = -1
                for i, det_box in enumerate(detections.xyxy):
                    iou = self._compute_iou(seed_box_arr, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_iou > 0.3 and best_idx >= 0:  # Minimum IoU threshold
                    cluster_id = predictions[best_idx]
                    cluster_votes[cluster_id].append(team_name)
                    print(f"  Seed match: {team_name} box -> cluster {cluster_id} (IoU={best_iou:.2f})")
                else:
                    print(f"  Warning: No match found for {team_name} seed box (best IoU={best_iou:.2f})")

        # Determine mapping based on votes
        team1_in_cluster_0 = cluster_votes[0].count(team1_name)
        team1_in_cluster_1 = cluster_votes[1].count(team1_name)
        team2_in_cluster_0 = cluster_votes[0].count(team2_name)
        team2_in_cluster_1 = cluster_votes[1].count(team2_name)

        print(f"  Cluster 0: {team1_in_cluster_0}x {team1_name}, {team2_in_cluster_0}x {team2_name}")
        print(f"  Cluster 1: {team1_in_cluster_1}x {team1_name}, {team2_in_cluster_1}x {team2_name}")

        # Score each possible mapping
        # Mapping A: cluster 0 = team1, cluster 1 = team2
        score_a = team1_in_cluster_0 + team2_in_cluster_1
        # Mapping B: cluster 0 = team2, cluster 1 = team1
        score_b = team2_in_cluster_0 + team1_in_cluster_1

        if score_a >= score_b:
            print(f"  Using mapping: cluster 0 = {team1_name}, cluster 1 = {team2_name}")
            return team1_name, team2_name
        else:
            print(f"  Using mapping: cluster 0 = {team2_name}, cluster 1 = {team1_name}")
            return team2_name, team1_name

    def analyze(
        self,
        video_path: Path,
        team1_name: Optional[str] = None,
        team2_name: Optional[str] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> AnalysisResult:
        """Run full analysis on a basketball video.

        Args:
            video_path: Path to input video
            team1_name: Name of first team
            team2_name: Name of second team
            progress_tracker: Optional ProgressTracker for UI updates

        Returns:
            AnalysisResult with paths to generated outputs
        """
        self._progress_tracker = progress_tracker
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
        total_frames = video_processor.total_frames

        # Set up progress phases if tracker provided
        if self._progress_tracker:
            self._progress_tracker.set_phases([
                ("Collecting samples", 0.05),
                ("Forward", 0.50),
                ("Backward", 0.15),
                ("Generating annotated video", 0.20),
                ("Generating court map video", 0.10),
            ])

        # Initialize shot detector
        if not self.config.skip_shot_detection:
            self._shot_detector = ShotDetector(fps=fps)

        # Phase 1: Collect training samples and train team classifier
        crops = self._collect_training_samples(video_processor)
        if len(crops) > 0:
            teams = self._train_team_classifier(crops)

            # Determine correct team mapping using seed annotations (if provided)
            # This fixes the cluster-to-team assignment that K-means can't determine
            cluster0_team, cluster1_team = self._determine_team_mapping_from_seeds(
                video_processor, team1, team2
            )
            self._setup_team_mapping(cluster0_team, cluster1_team)
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

        # Mark progress complete
        if self._progress_tracker:
            self._progress_tracker.complete()

        print(f"Analysis complete! Outputs saved to: {output_dir}")
        return result

    def _run_tracking_pass(
        self,
        video_processor: VideoProcessor,
    ) -> Tuple[List, List, np.ndarray, List]:
        """Run full tracking pass over video with bidirectional support.

        If reference_frame > 0, runs bidirectional tracking:
        - Forward pass: reference_frame → end
        - Backward pass: reference_frame → 0 (on reversed frames)
        - Merge: Combines both passes for complete video coverage

        Returns:
            Tuple of (frames, detections, teams, court_positions)
        """
        # Validators (shared across both passes)
        number_validator = ConsecutiveValueTracker(n_consecutive=3)
        team_validator = ConsecutiveValueTracker(n_consecutive=1)

        # === Reference Frame Selection ===
        reference_frame_idx = self.config.reference_frame
        if reference_frame_idx >= video_processor.total_frames:
            print(f"Warning: reference_frame {reference_frame_idx} exceeds video length, using frame 0")
            reference_frame_idx = 0

        # Get reference frame for initialization
        ref_frame_generator = video_processor.get_frames(start=reference_frame_idx)
        reference_frame = next(ref_frame_generator)

        # Initial detection on reference frame
        initial_detections = self._detector.detect_players(reference_frame)
        initial_detections.tracker_id = np.arange(1, len(initial_detections) + 1)
        print(f"Detected {len(initial_detections)} players in reference frame {reference_frame_idx}")

        # Get team assignments for reference frame
        boxes = sv.scale_boxes(xyxy=initial_detections.xyxy, factor=0.4)
        crops = [sv.crop_image(reference_frame, box) for box in boxes]
        teams = self._team_classifier.predict(crops)
        team_validator.update(tracker_ids=initial_detections.tracker_id.tolist(), values=teams.tolist())

        # Decide tracking strategy
        if reference_frame_idx == 0:
            # Simple forward-only tracking (original behavior)
            print("Running forward-only tracking from frame 0...")
            return self._run_forward_pass(
                video_processor,
                reference_frame,
                initial_detections,
                reference_frame_idx,
                teams,
                number_validator,
                team_validator,
            )
        else:
            # Bidirectional tracking
            print(f"Running bidirectional tracking from reference frame {reference_frame_idx}...")
            return self._run_bidirectional_pass(
                video_processor,
                reference_frame,
                initial_detections,
                reference_frame_idx,
                teams,
                number_validator,
                team_validator,
            )

    def _run_forward_pass(
        self,
        video_processor: VideoProcessor,
        reference_frame: np.ndarray,
        initial_detections: sv.Detections,
        reference_frame_idx: int,
        teams: np.ndarray,
        number_validator: ConsecutiveValueTracker,
        team_validator: ConsecutiveValueTracker,
    ) -> Tuple[List, List, np.ndarray, List]:
        """Run forward tracking pass from reference frame to end.

        Returns:
            Tuple of (frames, detections, teams, court_positions)
        """
        frames_history = []
        detections_history = []
        court_positions_history = []

        # Initialize tracker
        self._tracker.prompt_first_frame(reference_frame, initial_detections)

        # Process frames from reference frame onwards
        total_frames_to_process = video_processor.total_frames - reference_frame_idx
        frame_generator = video_processor.get_frames(start=reference_frame_idx)

        if self._progress_tracker:
            self._progress_tracker.start_phase("Forward", total=total_frames_to_process)

        print(f"  Forward pass: {total_frames_to_process} frames")

        for local_idx, frame in tqdm(enumerate(frame_generator), total=total_frames_to_process, desc="Forward"):
            frame_idx = reference_frame_idx + local_idx

            frames_history.append(frame)

            # Track
            player_detections = self._tracker.propagate(frame)
            detections_history.append(player_detections)

            # OCR (at intervals)
            if not self.config.skip_ocr and frame_idx % self.config.ocr_interval == 0:
                self._process_jersey_numbers(frame, player_detections, number_validator)

            # Court mapping
            if not self.config.skip_court_map:
                court_xy = self._process_court_positions(frame, player_detections)
                court_positions_history.append(court_xy)

            # Shot detection
            if not self.config.skip_shot_detection:
                shot_events = self._detector.detect_shot_events(frame)
                self._shot_detector.update(frame_idx=frame_idx, **shot_events)

            # Update progress
            if self._progress_tracker:
                self._progress_tracker.set_step(local_idx + 1)

        return frames_history, detections_history, teams, court_positions_history

    def _run_backward_pass(
        self,
        video_processor: VideoProcessor,
        reference_frame: np.ndarray,
        initial_detections: sv.Detections,
        reference_frame_idx: int,
        number_validator: ConsecutiveValueTracker,
    ) -> Tuple[List, List, List]:
        """Run backward tracking pass from reference frame to start.

        Loads frames 0 to reference_frame, reverses them, and tracks.
        Returns results in ORIGINAL (forward) order.

        Returns:
            Tuple of (frames, detections, court_positions) in forward order
        """
        if reference_frame_idx <= 0:
            return [], [], []

        if self._progress_tracker:
            self._progress_tracker.start_phase("Backward", total=reference_frame_idx + 1)

        print(f"  Backward pass: {reference_frame_idx} frames (loading into memory...)")

        # Load frames from 0 to reference_frame (inclusive) into memory
        frames_to_reverse = []
        frame_generator = video_processor.get_frames(start=0, end=reference_frame_idx + 1)
        load_idx = 0
        for frame in tqdm(frame_generator, total=reference_frame_idx + 1, desc="Loading"):
            frames_to_reverse.append(frame)
            load_idx += 1
            if self._progress_tracker:
                self._progress_tracker.set_step(load_idx // 2)  # Loading is half of backward phase

        # Reverse the frames: [0, 1, ..., ref] → [ref, ref-1, ..., 0]
        reversed_frames = frames_to_reverse[::-1]

        # Create new tracker for backward pass (same initialization)
        from hoopmania.models import SAM2Tracker
        backward_tracker = SAM2Tracker()

        # Initialize with reference frame (first in reversed list)
        # Use same detections to maintain consistent tracker IDs
        backward_tracker.prompt_first_frame(reversed_frames[0], initial_detections)

        # Track through reversed frames
        backward_frames = []
        backward_detections = []
        backward_court_positions = []

        print(f"  Tracking backward through {len(reversed_frames)} frames...")

        half_total = (reference_frame_idx + 1) // 2
        for local_idx, frame in tqdm(enumerate(reversed_frames), total=len(reversed_frames), desc="Backward"):
            # Original frame index (in forward order)
            original_frame_idx = reference_frame_idx - local_idx

            backward_frames.append(frame)

            # Track
            player_detections = backward_tracker.propagate(frame)
            backward_detections.append(player_detections)

            # OCR (at intervals) - use original frame index for interval check
            if not self.config.skip_ocr and original_frame_idx % self.config.ocr_interval == 0:
                self._process_jersey_numbers(frame, player_detections, number_validator)

            # Court mapping
            if not self.config.skip_court_map:
                court_xy = self._process_court_positions(frame, player_detections)
                backward_court_positions.append(court_xy)

            # Update progress (second half of backward phase)
            if self._progress_tracker:
                self._progress_tracker.set_step(half_total + local_idx + 1)

            # Note: Shot detection uses frame_idx for timing, which would be wrong
            # for reversed playback. We skip it here and rely on forward pass.

        # Reverse results back to forward order: [ref, ref-1, ..., 0] → [0, 1, ..., ref]
        backward_frames = backward_frames[::-1]
        backward_detections = backward_detections[::-1]
        backward_court_positions = backward_court_positions[::-1]

        return backward_frames, backward_detections, backward_court_positions

    def _run_bidirectional_pass(
        self,
        video_processor: VideoProcessor,
        reference_frame: np.ndarray,
        initial_detections: sv.Detections,
        reference_frame_idx: int,
        teams: np.ndarray,
        number_validator: ConsecutiveValueTracker,
        team_validator: ConsecutiveValueTracker,
    ) -> Tuple[List, List, np.ndarray, List]:
        """Run bidirectional tracking: forward and backward from reference frame.

        Returns:
            Tuple of (frames, detections, teams, court_positions) for complete video
        """
        # === Backward Pass (reference → 0) ===
        backward_frames, backward_detections, backward_court_positions = self._run_backward_pass(
            video_processor,
            reference_frame,
            initial_detections,
            reference_frame_idx,
            number_validator,
        )

        # === Forward Pass (reference → end) ===
        # Reset main tracker for forward pass
        self._tracker.reset()

        forward_frames, forward_detections, _, forward_court_positions = self._run_forward_pass(
            video_processor,
            reference_frame,
            initial_detections,
            reference_frame_idx,
            teams,
            number_validator,
            team_validator,
        )

        # === Merge Results ===
        # backward_results: [0, 1, ..., ref] (includes reference frame)
        # forward_results: [ref, ref+1, ..., end] (includes reference frame)
        # We need: [0, 1, ..., ref-1] + [ref, ref+1, ..., end]
        # So we exclude the last frame from backward (which is the reference frame)

        if len(backward_frames) > 0:
            # Exclude reference frame from backward (it's the last one after reversal)
            backward_frames_merged = backward_frames[:-1]
            backward_detections_merged = backward_detections[:-1]
            backward_court_merged = backward_court_positions[:-1] if backward_court_positions else []
        else:
            backward_frames_merged = []
            backward_detections_merged = []
            backward_court_merged = []

        # Merge: backward (0..ref-1) + forward (ref..end)
        frames_history = backward_frames_merged + forward_frames
        detections_history = backward_detections_merged + forward_detections

        if not self.config.skip_court_map:
            court_positions_history = backward_court_merged + forward_court_positions
        else:
            court_positions_history = []

        print(f"  Merged: {len(backward_frames_merged)} backward + {len(forward_frames)} forward = {len(frames_history)} total frames")

        return frames_history, detections_history, teams, court_positions_history

    def _get_valid_jersey_numbers(self) -> Optional[set]:
        """Get combined set of valid jersey numbers from both teams.

        Returns:
            Set of valid jersey number strings, or None if no whitelist configured
        """
        if not self.config.team1_jersey_numbers and not self.config.team2_jersey_numbers:
            return None

        valid = set()
        if self.config.team1_jersey_numbers:
            valid.update(str(n) for n in self.config.team1_jersey_numbers)
        if self.config.team2_jersey_numbers:
            valid.update(str(n) for n in self.config.team2_jersey_numbers)
        return valid

    def _filter_jersey_number(self, ocr_result: str, valid_numbers: Optional[set]) -> Optional[str]:
        """Filter/validate an OCR result against valid jersey numbers.

        Args:
            ocr_result: Raw OCR output string
            valid_numbers: Set of valid jersey number strings, or None to skip filtering

        Returns:
            Validated number string, or None if invalid/filtered
        """
        if valid_numbers is None:
            return ocr_result

        # Clean the OCR result - extract just digits
        cleaned = ''.join(c for c in str(ocr_result) if c.isdigit())

        if not cleaned:
            return None

        # Direct match
        if cleaned in valid_numbers:
            return cleaned

        # Try common OCR corrections
        corrections = {
            '0': ['8', '6'],  # 0 can be misread as 8 or 6
            '1': ['7'],       # 1 can be misread as 7
            '6': ['8', '0'],  # 6 can be misread as 8 or 0
            '8': ['0', '6'],  # 8 can be misread as 0 or 6
            '5': ['6'],       # 5 can be misread as 6
        }

        # Try single-character corrections for 2-digit numbers
        if len(cleaned) == 2:
            for i, char in enumerate(cleaned):
                if char in corrections:
                    for replacement in corrections[char]:
                        candidate = cleaned[:i] + replacement + cleaned[i+1:]
                        if candidate in valid_numbers:
                            return candidate

        return None  # Not a valid number

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

        # Apply jersey number whitelist filtering
        valid_numbers = self._get_valid_jersey_numbers()
        if valid_numbers is not None:
            numbers = [
                self._filter_jersey_number(n, valid_numbers)
                for n in numbers
            ]
            # Log filtering stats occasionally
            valid_count = sum(1 for n in numbers if n is not None)
            if valid_count < len(numbers):
                filtered_count = len(numbers) - valid_count
                print(f"  Jersey OCR: {valid_count} valid, {filtered_count} filtered")

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

            # Filter out None values (invalid numbers)
            valid_pairs = [
                (tid, num) for tid, num in zip(matched_tracker_ids, matched_numbers)
                if num is not None
            ]

            if valid_pairs:
                valid_tracker_ids, valid_matched_numbers = zip(*valid_pairs)
                number_validator.update(
                    tracker_ids=list(valid_tracker_ids),
                    values=list(valid_matched_numbers),
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

        if self._progress_tracker:
            self._progress_tracker.start_phase("Generating annotated video", total=len(frames_history))

        with sv.VideoSink(str(output_path), video_processor.info) as sink:
            for frame_idx, (frame, detections) in tqdm(
                enumerate(zip(frames_history, detections_history)),
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

                # Update progress
                if self._progress_tracker:
                    self._progress_tracker.set_step(frame_idx + 1)

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

        if self._progress_tracker:
            self._progress_tracker.start_phase("Generating court map video", total=len(court_positions_history))

        with sv.VideoSink(str(output_path), court_video_info) as sink:
            for frame_idx, positions in tqdm(
                enumerate(court_positions_history),
                total=len(court_positions_history),
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

                # Update progress
                if self._progress_tracker:
                    self._progress_tracker.set_step(frame_idx + 1)

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
