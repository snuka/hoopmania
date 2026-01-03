"""Command-line interface for HoopMania."""

from pathlib import Path
from typing import Optional
import os
import subprocess
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from hoopmania.config import PROJECT_ROOT, SAM2_DIR, DATA_DIR, OUTPUTS_DIR

app = typer.Typer(
    name="hoopmania",
    help="Basketball AI: Detect, Track, and Identify Basketball Players",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    video: Path = typer.Argument(
        ...,
        help="Path to input basketball video",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir", "-o",
        help="Output directory for results",
    ),
    team1: str = typer.Option(
        "Boston Celtics",
        "--team1", "-t1",
        help="Name of first team",
    ),
    team2: str = typer.Option(
        "New York Knicks",
        "--team2", "-t2",
        help="Name of second team",
    ),
    skip_ocr: bool = typer.Option(
        False,
        "--skip-ocr",
        help="Skip jersey number recognition",
    ),
    skip_court_map: bool = typer.Option(
        False,
        "--skip-court-map",
        help="Skip court position mapping",
    ),
    skip_shots: bool = typer.Option(
        False,
        "--skip-shots",
        help="Skip shot detection",
    ),
):
    """Analyze a basketball video.

    Detects players, tracks them through the video, identifies teams,
    recognizes jersey numbers, and generates annotated outputs.
    """
    from hoopmania.pipeline import BasketballAnalyzer, AnalysisConfig

    console.print(Panel.fit(
        "[bold blue]HoopMania Basketball Analyzer[/bold blue]\n"
        f"Video: {video.name}",
        title="Starting Analysis",
    ))

    # Configure analysis
    config = AnalysisConfig(
        team1_name=team1,
        team2_name=team2,
        skip_ocr=skip_ocr,
        skip_court_map=skip_court_map,
        skip_shot_detection=skip_shots,
        output_dir=output_dir or OUTPUTS_DIR,
    )

    # Run analysis
    try:
        analyzer = BasketballAnalyzer(config)
        result = analyzer.analyze(video, team1_name=team1, team2_name=team2)

        console.print("\n[bold green]Analysis Complete![/bold green]\n")
        console.print("Generated outputs:")

        if result.annotated_video_path:
            console.print(f"  [blue]Annotated Video:[/blue] {result.annotated_video_path}")
        if result.court_map_video_path:
            console.print(f"  [blue]Court Map:[/blue] {result.court_map_video_path}")
        if result.shot_chart_path:
            console.print(f"  [blue]Shot Chart:[/blue] {result.shot_chart_path}")
        if result.tracking_data_path:
            console.print(f"  [blue]Tracking Data:[/blue] {result.tracking_data_path}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def setup():
    """Download and set up required models and checkpoints.

    This will:
    - Clone the SAM2 repository
    - Download SAM2 checkpoints
    - Download font files
    """
    console.print(Panel.fit(
        "[bold blue]HoopMania Setup[/bold blue]\n"
        "Setting up models and checkpoints...",
    ))

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "videos").mkdir(exist_ok=True)
    (DATA_DIR / "checkpoints").mkdir(exist_ok=True)
    (DATA_DIR / "fonts").mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clone SAM2
    if not SAM2_DIR.exists():
        console.print("\n[yellow]Cloning SAM2 repository...[/yellow]")
        subprocess.run(
            ["git", "clone", "https://github.com/Gy920/segment-anything-2-real-time.git"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        console.print("\n[green]SAM2 repository already exists[/green]")

    # Install SAM2
    console.print("\n[yellow]Installing SAM2...[/yellow]")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=SAM2_DIR,
        check=True,
    )

    # Build Cython extensions
    console.print("\n[yellow]Building SAM2 extensions...[/yellow]")
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=SAM2_DIR,
        check=True,
    )

    # Download checkpoints
    checkpoints_dir = SAM2_DIR / "checkpoints"
    if not (checkpoints_dir / "sam2.1_hiera_large.pt").exists():
        console.print("\n[yellow]Downloading SAM2 checkpoints...[/yellow]")
        subprocess.run(
            ["bash", "download_ckpts.sh"],
            cwd=checkpoints_dir,
            check=True,
        )
    else:
        console.print("\n[green]SAM2 checkpoints already downloaded[/green]")

    # Download fonts
    fonts_dir = DATA_DIR / "fonts"
    if not (fonts_dir / "Staatliches-Regular.ttf").exists():
        console.print("\n[yellow]Downloading fonts...[/yellow]")
        try:
            subprocess.run(
                [
                    "gdown",
                    "https://drive.google.com/drive/folders/1RBjpI5Xleb58lujeusxH0W5zYMMA4ytO",
                    "-O", str(fonts_dir),
                    "--folder",
                ],
                check=True,
            )
        except Exception:
            console.print("[yellow]Could not download fonts. Labels may use default font.[/yellow]")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. Set your API keys in .env file:")
    console.print("     ROBOFLOW_API_KEY=your_key")
    console.print("     HF_TOKEN=your_token")
    console.print("  2. Run analysis:")
    console.print("     hoopmania analyze your_video.mp4")


@app.command()
def list_teams():
    """List available team configurations."""
    from hoopmania.config import TEAM_COLORS, TEAM_ROSTERS

    console.print("\n[bold]Available Teams:[/bold]\n")

    for team_name, color in sorted(TEAM_COLORS.items()):
        roster = TEAM_ROSTERS.get(team_name, {})
        has_roster = "Yes" if roster else "No"
        console.print(f"  [bold]{team_name}[/bold]")
        console.print(f"    Color: {color}")
        console.print(f"    Roster: {has_roster}")
        if roster:
            players = ", ".join([f"#{n} {p}" for n, p in list(roster.items())[:5]])
            console.print(f"    Players: {players}...")
        console.print()


@app.command()
def version():
    """Show version information."""
    from hoopmania import __version__

    console.print(f"HoopMania v{__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
