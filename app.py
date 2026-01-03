"""Gradio Web Interface for HoopMania Basketball Analyzer."""

import os
import sys
from pathlib import Path
import tempfile
import shutil

import gradio as gr

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variable for ONNX Runtime
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"


def get_available_teams():
    """Get list of available teams."""
    from hoopmania.config import TEAM_COLORS
    return list(TEAM_COLORS.keys())


def analyze_video(
    video_file,
    team1_name,
    team2_name,
    skip_ocr,
    skip_court_map,
    skip_shots,
    progress=gr.Progress()
):
    """Run basketball analysis on uploaded video."""
    if video_file is None:
        return None, None, None, "Please upload a video file."

    try:
        from hoopmania.pipeline import BasketballAnalyzer, AnalysisConfig
        from hoopmania.config import OUTPUTS_DIR

        progress(0.1, desc="Initializing models...")

        # Create output directory
        video_path = Path(video_file)
        output_dir = OUTPUTS_DIR / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configure analysis
        config = AnalysisConfig(
            team1_name=team1_name,
            team2_name=team2_name,
            skip_ocr=skip_ocr,
            skip_court_map=skip_court_map,
            skip_shot_detection=skip_shots,
            output_dir=OUTPUTS_DIR,
        )

        progress(0.2, desc="Running analysis...")

        # Run analysis
        analyzer = BasketballAnalyzer(config)
        result = analyzer.analyze(
            video_path,
            team1_name=team1_name,
            team2_name=team2_name,
        )

        progress(0.9, desc="Preparing outputs...")

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

        return annotated_video, court_map_video, shot_chart, status

    except Exception as e:
        import traceback
        error_msg = f"Error during analysis:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


def create_interface():
    """Create Gradio interface."""
    teams = get_available_teams()

    with gr.Blocks(
        title="HoopMania - Basketball AI",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .output-video { min-height: 400px; }
        """
    ) as app:
        gr.Markdown(
            """
            # 🏀 HoopMania - Basketball AI Analyzer

            Upload a basketball video to detect, track, and identify players.
            The analyzer will generate:
            - **Annotated Video**: Players with team colors, masks, and jersey numbers
            - **Court Map**: Bird's-eye view of player positions
            - **Shot Chart**: Made/missed shot locations

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Input")

                video_input = gr.Video(
                    label="Upload Basketball Video",
                    sources=["upload"],
                )

                gr.Markdown("### 🏀 Team Settings")

                team1_dropdown = gr.Dropdown(
                    choices=teams,
                    value="Boston Celtics",
                    label="Team 1 (Usually Home)",
                    allow_custom_value=True,
                )

                team2_dropdown = gr.Dropdown(
                    choices=teams,
                    value="New York Knicks",
                    label="Team 2 (Usually Away)",
                    allow_custom_value=True,
                )

                gr.Markdown("### ⚙️ Processing Options")

                skip_ocr = gr.Checkbox(
                    label="Skip Jersey Number Recognition (Faster)",
                    value=False,
                )

                skip_court = gr.Checkbox(
                    label="Skip Court Mapping",
                    value=False,
                )

                skip_shots = gr.Checkbox(
                    label="Skip Shot Detection",
                    value=False,
                )

                analyze_btn = gr.Button(
                    "🚀 Analyze Video",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Outputs")

                with gr.Tabs():
                    with gr.TabItem("Annotated Video"):
                        annotated_output = gr.Video(
                            label="Annotated Video",
                            elem_classes=["output-video"],
                        )

                    with gr.TabItem("Court Map"):
                        court_map_output = gr.Video(
                            label="Court Map Video",
                            elem_classes=["output-video"],
                        )

                    with gr.TabItem("Shot Chart"):
                        shot_chart_output = gr.Image(
                            label="Shot Chart",
                            type="filepath",
                        )

                status_output = gr.Textbox(
                    label="Status",
                    lines=6,
                    interactive=False,
                )

        # Connect analyze button
        analyze_btn.click(
            fn=analyze_video,
            inputs=[
                video_input,
                team1_dropdown,
                team2_dropdown,
                skip_ocr,
                skip_court,
                skip_shots,
            ],
            outputs=[
                annotated_output,
                court_map_output,
                shot_chart_output,
                status_output,
            ],
        )

        gr.Markdown(
            """
            ---
            ### 📝 Notes

            - **First run** may take longer as models are downloaded
            - **GPU recommended**: RTX 4090 or similar for best performance
            - **Processing time**: ~2-5 minutes per minute of video

            ### 🔧 Tips

            - Use **Skip Jersey Number Recognition** for 2x faster processing
            - For quick testing, use short video clips (10-30 seconds)
            - Team colors are assigned automatically based on jersey clustering

            ---
            Made with ❤️ using Roboflow, SAM2, and Gradio
            """
        )

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
