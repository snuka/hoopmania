# HoopMania - Basketball AI

Detect, track, and identify basketball players using computer vision and deep learning.

## Features

- **Player Detection**: RF-DETR based detection of players, referees, and ball
- **Player Tracking**: SAM2 based segmentation and tracking with stable IDs
- **Team Classification**: Automatic team assignment using SigLIP embeddings
- **Jersey Number Recognition**: SmolVLM2 based OCR for jersey numbers
- **Court Mapping**: Homography-based mapping to 2D court view
- **Shot Detection**: Detect made and missed shots

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 4090 recommended)
- ~15GB VRAM for full pipeline

## Installation

### 1. Clone and install

```bash
cd hoopmania
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Install additional dependencies

```bash
pip install git+https://github.com/roboflow/sports.git@feat/basketball
```

### 3. Set up SAM2

```bash
hoopmania setup
# Or manually:
bash scripts/setup_sam2.sh
```

### 4. Configure API keys

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your keys
```

Required keys:
- `ROBOFLOW_API_KEY`: Get from https://app.roboflow.com/settings/api
- `HF_TOKEN`: Get from https://huggingface.co/settings/tokens

## Usage

### Basic Analysis

```bash
# Full analysis with default settings
hoopmania analyze game.mp4

# Specify teams
hoopmania analyze game.mp4 --team1 "Boston Celtics" --team2 "New York Knicks"

# Output to specific directory
hoopmania analyze game.mp4 -o ./my_outputs
```

### Quick Mode (Skip OCR)

```bash
# Skip jersey number recognition for faster processing
hoopmania analyze game.mp4 --skip-ocr
```

### Minimal Mode

```bash
# Just detection and tracking
hoopmania analyze game.mp4 --skip-ocr --skip-court-map --skip-shots
```

### List Available Teams

```bash
hoopmania list-teams
```

## Outputs

The analyzer generates:

1. **Annotated Video** (`*_annotated.mp4`): Video with player masks, team colors, and labels
2. **Court Map Video** (`*_court_map.mp4`): Bird's-eye view of player positions
3. **Shot Chart** (`shot_chart.png`): Made/missed shot locations
4. **Tracking Data** (`tracking_data.json`): Full tracking data export

## Project Structure

```
hoopmania/
├── hoopmania/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration
│   ├── pipeline.py         # Main analysis pipeline
│   ├── models/             # Model wrappers
│   │   ├── detector.py     # RF-DETR player detection
│   │   ├── tracker.py      # SAM2 tracking
│   │   ├── classifier.py   # Team classification
│   │   ├── ocr.py          # Jersey number OCR
│   │   └── keypoints.py    # Court keypoint detection
│   ├── processing/         # Processing utilities
│   │   ├── video.py        # Video I/O
│   │   ├── court.py        # Court mapping
│   │   ├── paths.py        # Path smoothing
│   │   └── shots.py        # Shot detection
│   ├── visualization/      # Visualization
│   │   ├── annotators.py   # Frame annotators
│   │   └── court_renderer.py # Court visualization
│   └── utils/              # Utilities
│       └── validators.py   # Number validation
├── scripts/
│   ├── setup_sam2.sh
│   └── download_checkpoints.sh
├── data/
│   ├── videos/
│   ├── checkpoints/
│   └── fonts/
└── outputs/
```

## GPU Memory Usage

| Component | Approximate VRAM |
|-----------|-----------------|
| RF-DETR | ~2GB |
| SAM2 Large | ~8GB |
| SigLIP | ~2GB |
| SmolVLM2 | ~2GB |
| **Total** | **~14GB** |

## Troubleshooting

### Out of Memory

- Use `--skip-ocr` to skip jersey number recognition
- Process shorter video clips
- Consider using SAM2 small/medium checkpoints

### Model Download Issues

Ensure your API keys are correctly set:

```bash
echo $ROBOFLOW_API_KEY
echo $HF_TOKEN
```

### SAM2 Build Errors

Make sure you have build tools installed:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

## Credits

Based on the Roboflow basketball AI notebook:
- [Basketball AI Tutorial](https://blog.roboflow.com/identify-basketball-players)
- [Roboflow Notebooks](https://github.com/roboflow/notebooks)

## License

MIT License
