"""Frame extraction and annotation utilities for interactive UI."""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Generator
import numpy as np
import cv2
import supervision as sv


# === Frame Quality Scoring ===

def score_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    target_player_count: int = 10,
) -> Dict:
    """Score a frame for reference frame quality.

    Args:
        frame: RGB image
        detections: Player detections for this frame
        target_player_count: Expected number of players (default 10 for basketball)

    Returns:
        Dict with scoring components and combined score
    """
    player_count = len(detections)

    # Player count score (0-1, 1 = exactly target, decreases as we deviate)
    if target_player_count > 0:
        count_diff = abs(player_count - target_player_count)
        count_score = max(0, 1.0 - count_diff / target_player_count)
    else:
        count_score = 1.0 if player_count > 0 else 0.0

    # Spread score (higher = players more spread out, not clustered)
    if player_count >= 2:
        # Calculate center of each detection
        centers = (detections.xyxy[:, :2] + detections.xyxy[:, 2:]) / 2
        # Standard deviation of positions indicates spread
        spread_std = np.std(centers, axis=0).mean()
        # Normalize by frame diagonal
        frame_diag = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
        spread_score = min(1.0, spread_std / (frame_diag * 0.2))
    else:
        spread_score = 0.0

    # Confidence score (average detection confidence)
    if detections.confidence is not None and len(detections) > 0:
        confidence_score = float(np.mean(detections.confidence))
    else:
        confidence_score = 0.5

    # Size consistency score (players should be similar sizes, not one huge close-up)
    if player_count >= 2:
        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        area_std = np.std(areas) / (np.mean(areas) + 1e-6)
        size_consistency_score = max(0, 1.0 - area_std)
    else:
        size_consistency_score = 0.5

    # Combined score (weighted average)
    combined = (
        0.35 * count_score +
        0.25 * spread_score +
        0.25 * confidence_score +
        0.15 * size_consistency_score
    )

    return {
        "player_count": player_count,
        "count_score": count_score,
        "spread_score": spread_score,
        "confidence_score": confidence_score,
        "size_consistency_score": size_consistency_score,
        "combined_score": combined,
    }


def find_best_reference_frame(
    video_path: str,
    detector,  # PlayerDetector instance
    sample_interval: int = 15,
    max_seconds: float = 20.0,
    target_player_count: int = 10,
    progress_callback=None,
) -> Tuple[int, Optional[np.ndarray], Optional[sv.Detections], Dict]:
    """Find the best reference frame in a video automatically.

    Samples frames from the beginning of the video and scores them
    based on player count, spread, and detection confidence.

    Args:
        video_path: Path to video file
        detector: PlayerDetector instance for detection
        sample_interval: Sample every N frames
        max_seconds: Maximum seconds into video to search
        target_player_count: Expected number of players
        progress_callback: Optional callback(current, total, message) for progress

    Returns:
        Tuple of (best_frame_idx, best_frame_rgb, best_detections, score_info)
    """
    info = get_video_info(video_path)
    max_frame_idx = min(
        info['total_frames'] - 1,
        int(max_seconds * info['fps'])
    )

    best_frame_idx = 0
    best_frame = None
    best_detections = None
    best_score = -1
    best_score_info = {"combined_score": 0, "player_count": 0}

    # Calculate total samples for progress
    sample_indices = list(range(0, max_frame_idx, sample_interval))
    total_samples = len(sample_indices)

    for i, frame_idx in enumerate(sample_indices):
        if progress_callback:
            progress_callback(i, total_samples, f"Scoring frame {frame_idx}...")

        frame_rgb = extract_frame(video_path, frame_idx)
        if frame_rgb is None:
            continue

        # Convert to BGR for detector
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections = detector.detect_players(frame_bgr)

        # Score this frame
        score_info = score_frame(frame_rgb, detections, target_player_count)

        if score_info["combined_score"] > best_score:
            best_score = score_info["combined_score"]
            best_frame_idx = frame_idx
            best_frame = frame_rgb
            best_detections = detections
            best_score_info = score_info

    if progress_callback:
        progress_callback(total_samples, total_samples, "Auto-selection complete!")

    return best_frame_idx, best_frame, best_detections, best_score_info


# === Frame Extraction ===

def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a single frame from video.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract

    Returns:
        BGR frame as numpy array, or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB for display
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    except Exception:
        return None


def get_video_info(video_path: str) -> dict:
    """Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dict with fps, total_frames, width, height
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def extract_frame_thumbnails(
    video_path: str,
    num_thumbnails: int = 10,
    thumbnail_width: int = 160,
) -> List[Tuple[int, np.ndarray]]:
    """Extract evenly-spaced thumbnail frames from video.

    Args:
        video_path: Path to video file
        num_thumbnails: Number of thumbnails to extract
        thumbnail_width: Width of each thumbnail

    Returns:
        List of (frame_idx, thumbnail_image) tuples
    """
    info = get_video_info(video_path)
    total_frames = info["total_frames"]

    if total_frames <= 0:
        return []

    # Calculate frame indices
    indices = np.linspace(0, total_frames - 1, num_thumbnails, dtype=int)

    thumbnails = []
    cap = cv2.VideoCapture(video_path)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize to thumbnail
            aspect = frame.shape[0] / frame.shape[1]
            thumb_height = int(thumbnail_width * aspect)
            thumb = cv2.resize(frame, (thumbnail_width, thumb_height))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            thumbnails.append((int(idx), thumb_rgb))

    cap.release()
    return thumbnails


def draw_detections_with_ids(
    frame: np.ndarray,
    detections: sv.Detections,
    selected_team1: List[int] = None,
    selected_team2: List[int] = None,
    team1_color: Tuple[int, int, int] = (0, 122, 51),  # Green
    team2_color: Tuple[int, int, int] = (0, 107, 182),  # Blue
    unselected_color: Tuple[int, int, int] = (128, 128, 128),  # Gray
) -> np.ndarray:
    """Draw detection boxes with ID numbers for team assignment.

    Args:
        frame: RGB image
        detections: Player detections
        selected_team1: List of detection indices assigned to team 1
        selected_team2: List of detection indices assigned to team 2
        team1_color: RGB color for team 1
        team2_color: RGB color for team 2
        unselected_color: RGB color for unassigned players

    Returns:
        Annotated frame
    """
    selected_team1 = selected_team1 or []
    selected_team2 = selected_team2 or []

    annotated = frame.copy()

    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Determine color based on team assignment
        if i in selected_team1:
            color = team1_color
            label = f"T1-{i}"
        elif i in selected_team2:
            color = team2_color
            label = f"T2-{i}"
        else:
            color = unselected_color
            label = f"{i}"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Draw clickable center marker
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 8, color, -1)
        cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 2)

    return annotated


def find_clicked_detection(
    click_x: float,
    click_y: float,
    image_display_width: int,
    image_display_height: int,
    original_width: int,
    original_height: int,
    detections: sv.Detections,
) -> Optional[int]:
    """Find which detection was clicked.

    Args:
        click_x, click_y: Click coordinates in display space
        image_display_width/height: Size of displayed image
        original_width/height: Size of original image
        detections: Player detections

    Returns:
        Index of clicked detection, or None
    """
    # Scale click coordinates to original image space
    scale_x = original_width / image_display_width
    scale_y = original_height / image_display_height

    orig_x = click_x * scale_x
    orig_y = click_y * scale_y

    # Find detection containing the click point
    for i, box in enumerate(detections.xyxy):
        x1, y1, x2, y2 = box
        if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
            return i

    return None


def detections_to_seeds(
    detections: sv.Detections,
    team1_indices: List[int],
    team2_indices: List[int],
    team1_name: str,
    team2_name: str,
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Convert detection indices to team seeds format.

    Args:
        detections: Player detections
        team1_indices: Indices of players on team 1
        team2_indices: Indices of players on team 2
        team1_name: Name of team 1
        team2_name: Name of team 2

    Returns:
        Team seeds dict for AnalysisConfig
    """
    seeds = {}

    if team1_indices:
        seeds[team1_name] = [
            tuple(detections.xyxy[i].tolist())
            for i in team1_indices
            if i < len(detections.xyxy)
        ]

    if team2_indices:
        seeds[team2_name] = [
            tuple(detections.xyxy[i].tolist())
            for i in team2_indices
            if i < len(detections.xyxy)
        ]

    return seeds if seeds else None
