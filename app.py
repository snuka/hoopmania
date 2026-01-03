"""Gradio Web Interface for HoopMania Basketball Analyzer - Interactive UI."""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variable for ONNX Runtime
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"


# === Log Capture System ===
class LogCapture:
    """Thread-safe log capture for displaying in UI."""

    def __init__(self, max_lines=500):
        self.logs = []
        self.max_lines = max_lines
        self.lock = Lock()

    def add(self, message: str):
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message}")
            if len(self.logs) > self.max_lines:
                self.logs = self.logs[-self.max_lines:]

    def get_logs(self) -> str:
        with self.lock:
            return "\n".join(self.logs)

    def clear(self):
        with self.lock:
            self.logs = []


class LogHandler(logging.Handler):
    def __init__(self, log_capture: LogCapture):
        super().__init__()
        self.log_capture = log_capture

    def emit(self, record):
        msg = self.format(record)
        self.log_capture.add(msg)


log_capture = LogCapture()


def setup_logging():
    handler = LogHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    log_capture.add("=== HoopMania Log Viewer Started ===")


setup_logging()


# === Helper Functions ===
def get_available_teams():
    from hoopmania.config import TEAM_COLORS
    return list(TEAM_COLORS.keys())


def parse_jersey_numbers(numbers_str: str) -> Optional[set]:
    """Parse comma-separated jersey numbers into a set."""
    if not numbers_str or not numbers_str.strip():
        return None
    numbers = set()
    for part in numbers_str.split(","):
        part = part.strip()
        if part:
            numbers.add(part)
    return numbers if numbers else None


# === Frame Selection Functions ===
def on_video_upload(video_file):
    """Handle video upload - extract info and prepare frame selector."""
    if video_file is None:
        return (
            gr.update(visible=False),  # frame_selector_group
            gr.update(maximum=0, value=0),  # frame_slider
            None,  # frame_preview
            "",  # video_info_text
            None,  # state: video_path
            None,  # state: video_info
        )

    from hoopmania.utils.frame_selector import get_video_info, extract_frame

    video_path = video_file
    info = get_video_info(video_path)

    # Extract first frame for preview
    first_frame = extract_frame(video_path, 0)

    info_text = f"Duration: {info['total_frames']/info['fps']:.1f}s | {info['total_frames']} frames | {info['fps']:.1f} FPS | {info['width']}x{info['height']}"

    return (
        gr.update(visible=True),  # frame_selector_group
        gr.update(maximum=info['total_frames'] - 1, value=0),  # frame_slider
        first_frame,  # frame_preview
        info_text,  # video_info_text
        video_path,  # state: video_path
        info,  # state: video_info
    )


def on_auto_select_frame(video_path, video_info):
    """Automatically find the best reference frame."""
    if video_path is None:
        return (
            gr.update(),  # frame_slider
            None,  # frame_preview
            "Please upload a video first.",  # auto_select_status
        )

    log_capture.add("Auto-selecting best reference frame...")

    from hoopmania.utils.frame_selector import find_best_reference_frame
    from hoopmania.models import PlayerDetector

    # Initialize detector
    detector = PlayerDetector()

    # Find best frame
    best_idx, best_frame, best_detections, score_info = find_best_reference_frame(
        video_path,
        detector,
        sample_interval=15,
        max_seconds=20.0,
        target_player_count=10,
    )

    log_capture.add(f"Auto-selected frame {best_idx} with score {score_info['combined_score']:.2f}")
    log_capture.add(f"  Players: {score_info['player_count']}, Spread: {score_info['spread_score']:.2f}, Confidence: {score_info['confidence_score']:.2f}")

    status = f"""**Auto-selected frame {best_idx}**
- Players detected: {score_info['player_count']}
- Quality score: {score_info['combined_score']:.0%}
- Player spread: {score_info['spread_score']:.0%}
- Detection confidence: {score_info['confidence_score']:.0%}"""

    return (
        gr.update(value=best_idx),  # frame_slider
        best_frame,  # frame_preview
        status,  # auto_select_status
    )


def on_frame_slider_change(frame_idx, video_path):
    """Update frame preview when slider changes."""
    if video_path is None:
        return None

    from hoopmania.utils.frame_selector import extract_frame
    frame = extract_frame(video_path, int(frame_idx))
    return frame


def on_confirm_frame(frame_idx, video_path, video_info, team1_name, team2_name):
    """Confirm selected frame and run player detection for team assignment."""
    if video_path is None:
        return (
            gr.update(visible=False),  # team_assignment_group
            None,  # annotated_frame
            [],  # state: detections
            [],  # state: team1_indices
            [],  # state: team2_indices
            int(frame_idx),  # state: selected_frame
            "Please upload a video first.",  # status
        )

    log_capture.add(f"Selected reference frame: {int(frame_idx)}")

    from hoopmania.utils.frame_selector import extract_frame, draw_detections_with_ids
    from hoopmania.models import PlayerDetector

    # Extract frame
    frame = extract_frame(video_path, int(frame_idx))
    if frame is None:
        return (
            gr.update(visible=False),
            None, [], [], [], int(frame_idx),
            "Failed to extract frame.",
        )

    # Run player detection
    log_capture.add("Running player detection on reference frame...")
    detector = PlayerDetector()

    # Convert RGB to BGR for detection (detector expects BGR)
    import cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    detections = detector.detect_players(frame_bgr)

    log_capture.add(f"Detected {len(detections)} players")

    # Draw detections with IDs
    annotated = draw_detections_with_ids(frame, detections)

    # Convert detections to serializable format
    det_list = detections.xyxy.tolist() if len(detections) > 0 else []

    instructions = f"""
### Detected {len(detections)} Players

**Click on players to assign them to teams:**
- First, click players on **{team1_name}** (will turn green)
- Then toggle to **{team2_name}** mode and click those players (will turn blue)
- Click a player again to unassign them

Use the buttons below to switch between team assignment modes.
"""

    return (
        gr.update(visible=True),  # team_assignment_group
        annotated,  # annotated_frame
        det_list,  # state: detections
        [],  # state: team1_indices
        [],  # state: team2_indices
        int(frame_idx),  # state: selected_frame
        instructions,  # status
    )


def on_image_click(
    evt: gr.SelectData,
    frame_idx,
    video_path,
    detections_list,
    team1_indices,
    team2_indices,
    current_team_mode,
    team1_name,
    team2_name,
):
    """Handle click on annotated frame to toggle team assignment."""
    if video_path is None or not detections_list:
        return None, team1_indices, team2_indices, "No detections available."

    import supervision as sv
    from hoopmania.utils.frame_selector import extract_frame, draw_detections_with_ids

    # Get click coordinates
    click_x, click_y = evt.index

    # Convert detections list back to numpy
    detections_xyxy = np.array(detections_list)

    # Find which detection was clicked
    clicked_idx = None
    for i, box in enumerate(detections_xyxy):
        x1, y1, x2, y2 = box
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            clicked_idx = i
            break

    if clicked_idx is None:
        # Check if click was near center of any detection
        for i, box in enumerate(detections_xyxy):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
            if dist < 50:  # Within 50 pixels of center
                clicked_idx = i
                break

    if clicked_idx is None:
        return None, team1_indices, team2_indices, "Click on a player to assign them."

    # Toggle assignment
    team1_indices = list(team1_indices) if team1_indices else []
    team2_indices = list(team2_indices) if team2_indices else []

    if clicked_idx in team1_indices:
        team1_indices.remove(clicked_idx)
        action = f"Removed player {clicked_idx} from {team1_name}"
    elif clicked_idx in team2_indices:
        team2_indices.remove(clicked_idx)
        action = f"Removed player {clicked_idx} from {team2_name}"
    elif current_team_mode == "team1":
        team1_indices.append(clicked_idx)
        action = f"Assigned player {clicked_idx} to {team1_name}"
    else:
        team2_indices.append(clicked_idx)
        action = f"Assigned player {clicked_idx} to {team2_name}"

    log_capture.add(action)

    # Redraw frame with updated assignments
    frame = extract_frame(video_path, int(frame_idx))

    # Create supervision Detections object for drawing
    detections = sv.Detections(xyxy=detections_xyxy)

    annotated = draw_detections_with_ids(
        frame, detections,
        selected_team1=team1_indices,
        selected_team2=team2_indices,
    )

    status = f"{team1_name}: {len(team1_indices)} players | {team2_name}: {len(team2_indices)} players"

    return annotated, team1_indices, team2_indices, status


def update_team_mode(mode):
    """Update which team is being assigned."""
    return mode


# === Progress State ===
class ProgressState:
    """Thread-safe progress state for UI updates."""

    def __init__(self):
        self.lock = Lock()
        self.progress_pct = 0.0
        self.eta_str = "--:--"
        self.gpu_pct = 0.0
        self.phase_name = ""
        self.phase_step = 0
        self.phase_total = 0

    def update(self, progress_pct: float, eta_str: str, gpu_pct: float):
        with self.lock:
            self.progress_pct = progress_pct
            self.eta_str = eta_str
            self.gpu_pct = gpu_pct

    def set_phase(self, name: str, step: int, total: int):
        with self.lock:
            self.phase_name = name
            self.phase_step = step
            self.phase_total = total

    def get_status(self) -> str:
        with self.lock:
            if self.phase_total > 0:
                phase_info = f"{self.phase_name}: {self.phase_step}/{self.phase_total}"
            else:
                phase_info = self.phase_name or "Initializing..."

            return f"Progress: {self.progress_pct:.1f}% | ETA: {self.eta_str} | GPU: {self.gpu_pct:.0f}% | {phase_info}"


progress_state = ProgressState()


# === Analysis Function ===
def run_analysis(
    video_path,
    team1_name,
    team2_name,
    skip_ocr,
    skip_court_map,
    skip_shots,
    selected_frame,
    detections_list,
    team1_indices,
    team2_indices,
    team1_numbers,
    team2_numbers,
    progress=gr.Progress()
):
    """Run basketball analysis with all configured options."""
    if video_path is None:
        return None, None, None, "Please upload a video file."

    try:
        log_capture.add(f"Starting analysis...")
        log_capture.add(f"Video: {video_path}")
        log_capture.add(f"Teams: {team1_name} vs {team2_name}")

        from hoopmania.pipeline import BasketballAnalyzer, AnalysisConfig
        from hoopmania.config import OUTPUTS_DIR
        from hoopmania.utils.frame_selector import detections_to_seeds
        from hoopmania.utils.progress import ProgressTracker
        import supervision as sv

        # Create progress tracker with callback that updates both gr.Progress and our state
        def progress_callback(pct, eta, gpu):
            progress_state.update(pct, eta, gpu)
            # Update Gradio progress (0-1 scale)
            try:
                progress(pct / 100.0, desc=f"{pct:.1f}% | ETA: {eta} | GPU: {gpu:.0f}%")
            except Exception:
                pass  # Gradio progress may fail if called outside request context

        tracker = ProgressTracker(callback=progress_callback, update_interval=0.3)

        log_capture.add("Initializing models...")

        # Create output directory
        video_file_path = Path(video_path)
        output_dir = OUTPUTS_DIR / video_file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse jersey numbers
        team1_jersey_numbers = parse_jersey_numbers(team1_numbers)
        team2_jersey_numbers = parse_jersey_numbers(team2_numbers)

        # Convert team assignments to seeds
        team_seeds = None
        if detections_list and (team1_indices or team2_indices):
            detections = sv.Detections(xyxy=np.array(detections_list))
            team_seeds = detections_to_seeds(
                detections,
                team1_indices or [],
                team2_indices or [],
                team1_name,
                team2_name,
            )
            log_capture.add(f"Team seeds from annotations: {team_seeds}")

        # Reference frame
        ref_frame = int(selected_frame) if selected_frame else 0
        if ref_frame > 0:
            log_capture.add(f"Using reference frame: {ref_frame}")

        # Log settings
        if team1_jersey_numbers:
            log_capture.add(f"{team1_name} jersey numbers: {team1_jersey_numbers}")
        if team2_jersey_numbers:
            log_capture.add(f"{team2_name} jersey numbers: {team2_jersey_numbers}")

        # Configure analysis
        config = AnalysisConfig(
            team1_name=team1_name,
            team2_name=team2_name,
            skip_ocr=skip_ocr,
            skip_court_map=skip_court_map,
            skip_shot_detection=skip_shots,
            output_dir=OUTPUTS_DIR,
            team1_jersey_numbers=team1_jersey_numbers,
            team2_jersey_numbers=team2_jersey_numbers,
            reference_frame=ref_frame,
            team_seeds=team_seeds,
        )

        log_capture.add("Running analysis pipeline...")

        # Run analysis with progress tracker
        analyzer = BasketballAnalyzer(config)
        result = analyzer.analyze(
            video_file_path,
            team1_name=team1_name,
            team2_name=team2_name,
            progress_tracker=tracker,
        )

        log_capture.add("Analysis complete, preparing outputs...")

        # Prepare outputs
        annotated_video = str(result.annotated_video_path) if result.annotated_video_path else None
        court_map_video = str(result.court_map_video_path) if result.court_map_video_path else None
        shot_chart = str(result.shot_chart_path) if result.shot_chart_path else None

        # Create status message
        status = "Analysis complete!\n\n"
        status += f"Outputs saved to: {output_dir}\n\n"

        if result.shot_events:
            status += f"Shot Events Detected: {len(result.shot_events)}\n"
            made = sum(1 for e in result.shot_events if e.get('made', False))
            status += f"  - Made: {made}\n"
            status += f"  - Missed: {len(result.shot_events) - made}\n"

        progress(1.0, desc="Done!")
        log_capture.add("=== Analysis Complete ===")

        return annotated_video, court_map_video, shot_chart, status

    except Exception as e:
        import traceback
        error_msg = f"Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
        log_capture.add(f"ERROR: {str(e)}")
        log_capture.add(traceback.format_exc())
        return None, None, None, error_msg


def refresh_logs():
    return log_capture.get_logs()


def clear_logs():
    log_capture.clear()
    log_capture.add("=== Logs Cleared ===")
    return log_capture.get_logs()


# === Main Interface ===
def create_interface():
    teams = get_available_teams()

    with gr.Blocks(title="HoopMania - Basketball AI") as app:
        # === State Variables ===
        video_path_state = gr.State(None)
        video_info_state = gr.State(None)
        detections_state = gr.State([])
        team1_indices_state = gr.State([])
        team2_indices_state = gr.State([])
        selected_frame_state = gr.State(0)
        current_team_mode = gr.State("team1")

        gr.Markdown(
            """
            # HoopMania - Basketball AI Analyzer

            Analyze basketball videos with AI-powered player detection, tracking, and team classification.

            **Workflow:** Upload Video -> Select Reference Frame -> Assign Teams -> Configure Options -> Analyze
            """
        )

        with gr.Row():
            # === Left Column: Setup ===
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Upload Video")
                video_input = gr.File(
                    label="Upload Basketball Video",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                )

                gr.Markdown("### Step 2: Team Settings")
                team1_dropdown = gr.Dropdown(
                    choices=teams,
                    value="Boston Celtics",
                    label="Team 1 (Home)",
                    allow_custom_value=True,
                )
                team2_dropdown = gr.Dropdown(
                    choices=teams,
                    value="New York Knicks",
                    label="Team 2 (Away)",
                    allow_custom_value=True,
                )

                gr.Markdown("### Step 3: Jersey Numbers (Optional)")
                team1_numbers = gr.Textbox(
                    label="Team 1 Jersey Numbers",
                    placeholder="e.g., 0, 7, 8, 11, 23, 32",
                    info="Comma-separated. Helps filter OCR errors.",
                )
                team2_numbers = gr.Textbox(
                    label="Team 2 Jersey Numbers",
                    placeholder="e.g., 0, 4, 9, 11, 20, 40",
                )

                gr.Markdown("### Step 4: Processing Options")
                skip_ocr = gr.Checkbox(label="Skip Jersey Number Recognition (Faster)", value=False)
                skip_court = gr.Checkbox(label="Skip Court Mapping", value=False)
                skip_shots = gr.Checkbox(label="Skip Shot Detection", value=False)

                analyze_btn = gr.Button("Analyze Video", variant="primary", size="lg")

            # === Middle Column: Frame Selection & Team Assignment ===
            with gr.Column(scale=2):
                gr.Markdown("### Reference Frame & Team Assignment")

                video_info_text = gr.Markdown("")

                # Frame selection group
                with gr.Group(visible=False) as frame_selector_group:
                    gr.Markdown("**Select a reference frame where all players are clearly visible:**")

                    with gr.Row():
                        auto_select_btn = gr.Button(
                            "Auto-Select Best Frame",
                            variant="primary",
                        )
                        manual_label = gr.Markdown("*or manually scrub below*")

                    auto_select_status = gr.Markdown("")

                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                        label="Frame (drag to scrub manually)",
                        interactive=True,
                    )
                    frame_preview = gr.Image(
                        label="Frame Preview",
                        elem_classes=["frame-preview"],
                        interactive=False,
                    )
                    confirm_frame_btn = gr.Button("Use This Frame for Team Assignment", variant="secondary")

                # Team assignment group
                with gr.Group(visible=False) as team_assignment_group:
                    assignment_status = gr.Markdown("")

                    with gr.Row():
                        team1_btn = gr.Button("Assign to Team 1", variant="primary", scale=1)
                        team2_btn = gr.Button("Assign to Team 2", variant="secondary", scale=1)
                        clear_assignments_btn = gr.Button("Clear All", variant="stop", scale=1)

                    annotated_frame = gr.Image(
                        label="Click on players to assign teams",
                        elem_classes=["frame-preview"],
                        interactive=True,
                    )

                    team_status = gr.Markdown("Click on players after selecting a team mode above.")

            # === Right Column: Results & Logs ===
            with gr.Column(scale=2):
                gr.Markdown("### Results")

                with gr.Tabs():
                    with gr.TabItem("Annotated Video"):
                        annotated_output = gr.Video(label="Annotated Video", elem_classes=["output-video"])
                    with gr.TabItem("Court Map"):
                        court_map_output = gr.Video(label="Court Map Video", elem_classes=["output-video"])
                    with gr.TabItem("Shot Chart"):
                        shot_chart_output = gr.Image(label="Shot Chart", type="filepath")

                status_output = gr.Textbox(label="Status", lines=4, interactive=False)

                gr.Markdown("### Logs")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")

                logs_output = gr.Textbox(
                    label="Backend Logs",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    elem_classes=["log-box"],
                    value=log_capture.get_logs(),
                )

                auto_refresh = gr.Timer(2)
                auto_refresh.tick(fn=refresh_logs, outputs=logs_output)

        # === Event Handlers ===

        # Video upload
        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[
                frame_selector_group,
                frame_slider,
                frame_preview,
                video_info_text,
                video_path_state,
                video_info_state,
            ],
        )

        # Frame slider change
        frame_slider.release(
            fn=on_frame_slider_change,
            inputs=[frame_slider, video_path_state],
            outputs=[frame_preview],
        )

        # Auto-select best frame
        auto_select_btn.click(
            fn=on_auto_select_frame,
            inputs=[video_path_state, video_info_state],
            outputs=[frame_slider, frame_preview, auto_select_status],
        )

        # Confirm frame button
        confirm_frame_btn.click(
            fn=on_confirm_frame,
            inputs=[frame_slider, video_path_state, video_info_state, team1_dropdown, team2_dropdown],
            outputs=[
                team_assignment_group,
                annotated_frame,
                detections_state,
                team1_indices_state,
                team2_indices_state,
                selected_frame_state,
                assignment_status,
            ],
        )

        # Team mode buttons
        team1_btn.click(
            fn=lambda: "team1",
            outputs=[current_team_mode],
        ).then(
            fn=lambda t1, t2: f"**Mode: Assigning to {t1}** - Click on players",
            inputs=[team1_dropdown, team2_dropdown],
            outputs=[team_status],
        )

        team2_btn.click(
            fn=lambda: "team2",
            outputs=[current_team_mode],
        ).then(
            fn=lambda t1, t2: f"**Mode: Assigning to {t2}** - Click on players",
            inputs=[team1_dropdown, team2_dropdown],
            outputs=[team_status],
        )

        # Clear assignments
        def clear_assignments(frame_idx, video_path, detections_list):
            if video_path is None or not detections_list:
                return None, [], [], "Assignments cleared."

            import supervision as sv
            from hoopmania.utils.frame_selector import extract_frame, draw_detections_with_ids

            frame = extract_frame(video_path, int(frame_idx))
            detections = sv.Detections(xyxy=np.array(detections_list))
            annotated = draw_detections_with_ids(frame, detections)

            return annotated, [], [], "All assignments cleared."

        clear_assignments_btn.click(
            fn=clear_assignments,
            inputs=[selected_frame_state, video_path_state, detections_state],
            outputs=[annotated_frame, team1_indices_state, team2_indices_state, team_status],
        )

        # Image click for team assignment
        annotated_frame.select(
            fn=on_image_click,
            inputs=[
                selected_frame_state,
                video_path_state,
                detections_state,
                team1_indices_state,
                team2_indices_state,
                current_team_mode,
                team1_dropdown,
                team2_dropdown,
            ],
            outputs=[annotated_frame, team1_indices_state, team2_indices_state, team_status],
        )

        # Analyze button
        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                video_path_state,
                team1_dropdown,
                team2_dropdown,
                skip_ocr,
                skip_court,
                skip_shots,
                selected_frame_state,
                detections_state,
                team1_indices_state,
                team2_indices_state,
                team1_numbers,
                team2_numbers,
            ],
            outputs=[annotated_output, court_map_output, shot_chart_output, status_output],
        )

        # Log buttons
        refresh_btn.click(fn=refresh_logs, outputs=logs_output)
        clear_btn.click(fn=clear_logs, outputs=logs_output)

        gr.Markdown(
            """
            ---
            ### Tips
            - **Reference Frame**: Choose a frame where all 10 players are visible and spread out
            - **Team Assignment**: Just assign 1-2 players per team to help the AI classify correctly
            - **Jersey Numbers**: Providing valid numbers helps filter OCR errors

            Made with Roboflow, SAM2, and Gradio
            """
        )

    return app


if __name__ == "__main__":
    app = create_interface()

    # Custom CSS for progress display
    custom_css = """
        .gradio-container { max-width: 1800px !important; }
        .output-video { min-height: 400px; }
        .log-box textarea { font-family: monospace !important; font-size: 12px !important; }
        .frame-preview img { max-height: 400px !important; object-fit: contain !important; }
        .team-btn-active { background-color: #4CAF50 !important; color: white !important; }
        .progress-box { font-family: monospace !important; font-size: 14px !important; background: #1a1a2e !important; color: #00ff88 !important; }
        """

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
