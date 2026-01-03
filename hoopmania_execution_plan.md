# HoopMania Execution Plan for Claude Code CLI

## Project Overview

Convert a Google Colab notebook into a standalone Python application that analyzes basketball game footage using computer vision and deep learning. The app will run locally on an RTX 4090.

---

## 🎯 End Goal

A CLI application that takes a basketball video as input and produces:
1. **Annotated video** with player detection, tracking, team colors, and jersey numbers
2. **2D court map video** showing player positions from bird's-eye view
3. **Shot chart** showing made/missed shot locations
4. **Player tracking data** (JSON/CSV export)

---

## 📦 Phase 1: Project Scaffolding

### Task 1.1: Create project structure
```
hoopmania/
├── hoopmania/
│   ├── __init__.py
│   ├── cli.py                    # Main entry point
│   ├── config.py                 # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py           # RF-DETR player detection
│   │   ├── tracker.py            # SAM2 tracking wrapper
│   │   ├── classifier.py         # Team classification (SigLIP)
│   │   ├── ocr.py                # Jersey number recognition (SmolVLM2)
│   │   └── keypoints.py          # Court keypoint detection
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── video.py              # Video I/O utilities
│   │   ├── court.py              # Court mapping & homography
│   │   ├── paths.py              # Path cleaning & smoothing
│   │   └── shots.py              # Shot event detection
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── annotators.py         # Custom annotators
│   │   └── court_renderer.py     # 2D court visualization
│   └── utils/
│       ├── __init__.py
│       └── validators.py         # Number validation, IoS matching
├── scripts/
│   ├── download_checkpoints.sh   # Download SAM2 checkpoints
│   └── setup_sam2.sh             # Clone and build SAM2
├── data/
│   ├── videos/                   # Input videos
│   ├── checkpoints/              # Model checkpoints
│   └── fonts/                    # Fonts for annotations
├── outputs/                      # Generated outputs
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

### Task 1.2: Create requirements.txt
```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
tqdm>=4.65.0

# Roboflow/Inference
inference-gpu>=0.9.0
supervision==0.27.0

# Sports utilities (Roboflow)
# Install from: git+https://github.com/roboflow/sports.git@feat/basketball

# Transformers for SigLIP and SmolVLM2
transformers>=4.35.0
num2words>=0.5.12

# Video processing
ffmpeg-python>=0.2.0

# CLI
typer>=0.9.0
rich>=13.0.0

# Environment
python-dotenv>=1.0.0

# Pinned for compatibility
cachetools==5.5.0
```

### Task 1.3: Create .env.example
```
ROBOFLOW_API_KEY=your_roboflow_api_key
HF_TOKEN=your_huggingface_token
```

---

## 📦 Phase 2: Model Setup & Download Scripts

### Task 2.1: Create SAM2 setup script (`scripts/setup_sam2.sh`)
- Clone `segment-anything-2-real-time` repo
- Install as editable package
- Build Cython extensions
- Download checkpoints (sam2.1_hiera_large.pt)

### Task 2.2: Create config.py
```python
# Model IDs
PLAYER_DETECTION_MODEL_ID = "rf-detr-basketball-4mvjl/1"
KEYPOINT_DETECTION_MODEL_ID = "basketball-court-detection-2/14"
NUMBER_RECOGNITION_MODEL_ID = "basketball-jersey-numbers-ocr/3"

# Confidence thresholds
PLAYER_DETECTION_CONFIDENCE = 0.3
PLAYER_DETECTION_IOU_THRESHOLD = 0.3
KEYPOINT_DETECTION_CONFIDENCE = 0.3
KEYPOINT_ANCHOR_CONFIDENCE = 0.5

# Class IDs
CLASS_IDS = {
    "ball": 0,
    "ball_in_basket": 1,
    "number": 2,
    "player": 3,
    "player_in_possession": 4,
    "player_jump_shot": 5,
    "player_layup_dunk": 6,
    "player_shot_block": 7,
    "referee": 8,
    "rim": 9
}

PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]

# Team colors
TEAM_COLORS = {
    "Boston Celtics": "#007A33",
    "New York Knicks": "#006BB6",
    # Add more teams...
}

# Team rosters (jersey number -> player name)
TEAM_ROSTERS = {
    "Boston Celtics": {
        "0": "Jayson Tatum",
        "7": "Jaylen Brown",
        # ...
    },
    # ...
}
```

---

## 📦 Phase 3: Core Model Wrappers

### Task 3.1: Create `models/detector.py`
**PlayerDetector class:**
- Load RF-DETR model from Roboflow
- `detect(frame)` → returns `sv.Detections`
- Filter by class IDs
- Configurable confidence/IOU thresholds

### Task 3.2: Create `models/tracker.py`
**SAM2Tracker class:**
- Initialize SAM2 camera predictor
- `prompt_first_frame(frame, detections)` - seed with RF-DETR boxes
- `propagate(frame)` → returns `sv.Detections` with masks and tracker IDs
- `reset()` - clear state for new video
- Handle mask filtering with `filter_segments_by_distance`

### Task 3.3: Create `models/classifier.py`
**TeamClassifier class:**
- Load SigLIP model from HuggingFace
- K-means clustering on embeddings (k=2)
- `fit(crops)` - train on first frame player crops
- `predict(crops)` → team IDs (0 or 1)

### Task 3.4: Create `models/ocr.py`
**JerseyOCR class:**
- Load SmolVLM2 from Roboflow
- `recognize(crop)` → jersey number string
- Batch processing support

### Task 3.5: Create `models/keypoints.py`
**CourtKeypointDetector class:**
- Load keypoint detection model
- `detect(frame)` → `sv.KeyPoints`
- Filter by confidence threshold

---

## 📦 Phase 4: Processing Pipeline

### Task 4.1: Create `processing/video.py`
- `get_frame_generator(video_path)`
- `get_video_info(video_path)`
- `process_video(source, target, callback)`
- `compress_video(input_path, output_path)` - ffmpeg wrapper

### Task 4.2: Create `processing/court.py`
**ViewTransformer wrapper:**
- `compute_homography(frame_landmarks, court_landmarks)`
- `transform_points(frame_xy)` → court_xy
- NBA court configuration (from `sports.basketball`)

### Task 4.3: Create `processing/paths.py`
**Path cleaning utilities:**
- `clean_paths()` - detect jumps, remove artifacts, interpolate, smooth
- Savitzky-Golay filter wrapper
- Jump detection with sigma threshold

### Task 4.4: Create `processing/shots.py`
**ShotEventTracker wrapper:**
- Track jump shots, layups, dunks
- Detect ball-in-basket events
- Mark made vs missed shots
- Return shot locations with timestamps

---

## 📦 Phase 5: Validation & Matching

### Task 5.1: Create `utils/validators.py`
**ConsecutiveValueTracker:**
- Track jersey numbers across frames
- Require N consecutive matches to confirm
- `update(tracker_ids, values)`
- `get_validated(tracker_ids)` → confirmed values

**IoS Matching:**
- `mask_ios_batch()` - intersection over smaller area
- `match_numbers_to_players(player_masks, number_boxes)` → pairs

---

## 📦 Phase 6: Visualization

### Task 6.1: Create `visualization/annotators.py`
**Custom annotators:**
- Team-colored mask annotator
- Team-colored box annotator
- Rich label annotator with custom fonts
- Jersey number + player name labels

### Task 6.2: Create `visualization/court_renderer.py`
**Court visualization:**
- `draw_court(config)` - NBA court template
- `draw_points_on_court(xy, colors)` - player positions
- `draw_paths_on_court(paths, colors)` - movement trails
- `draw_shot_chart(made_xy, missed_xy)` - shot locations

---

## 📦 Phase 7: Main Pipeline

### Task 7.1: Create `hoopmania/pipeline.py`
**BasketballAnalyzer class:**
```python
class BasketballAnalyzer:
    def __init__(self, config):
        self.detector = PlayerDetector(...)
        self.tracker = SAM2Tracker(...)
        self.team_classifier = TeamClassifier(...)
        self.jersey_ocr = JerseyOCR(...)
        self.keypoint_detector = CourtKeypointDetector(...)
        
    def analyze(self, video_path, output_dir):
        # Phase 1: Detection + Tracking
        # Phase 2: Team Classification  
        # Phase 3: Jersey Number Recognition
        # Phase 4: Court Mapping
        # Phase 5: Shot Detection
        # Phase 6: Generate Outputs
        pass
```

### Task 7.2: Create `cli.py`
```python
import typer
app = typer.Typer()

@app.command()
def analyze(
    video: Path,
    output_dir: Path = Path("outputs"),
    team1: str = "Team A",
    team2: str = "Team B",
    skip_ocr: bool = False,
    skip_court_map: bool = False,
):
    """Analyze a basketball video."""
    ...

@app.command()
def setup():
    """Download models and checkpoints."""
    ...

if __name__ == "__main__":
    app()
```

---

## 📦 Phase 8: Output Generation

### Task 8.1: Generate annotated video
- Player masks with team colors
- Jersey numbers + player names
- Bounding boxes
- Compress with ffmpeg

### Task 8.2: Generate court map video
- 2D bird's-eye view
- Player dots with team colors
- Synchronized with source video

### Task 8.3: Generate shot chart
- Static image of court
- Made shots (green circles)
- Missed shots (red X marks)
- Player-specific filtering option

### Task 8.4: Export tracking data
```json
{
  "video_info": {...},
  "frames": [
    {
      "frame_idx": 0,
      "players": [
        {
          "tracker_id": 1,
          "team": "Boston Celtics",
          "jersey_number": "0",
          "player_name": "Jayson Tatum",
          "bbox": [x1, y1, x2, y2],
          "court_position": [x, y]
        }
      ]
    }
  ],
  "shot_events": [...]
}
```

---

## 🚀 Execution Order for Claude Code

### Step 1: Initialize project
```bash
mkdir hoopmania && cd hoopmania
# Create all directories
# Create pyproject.toml and requirements.txt
```

### Step 2: Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/roboflow/sports.git@feat/basketball
```

### Step 3: SAM2 setup
```bash
git clone https://github.com/Gy920/segment-anything-2-real-time.git
cd segment-anything-2-real-time
pip install -e .
python setup.py build_ext --inplace
cd checkpoints && bash download_ckpts.sh
```

### Step 4: Implement modules (in order)
1. `config.py`
2. `models/detector.py`
3. `models/tracker.py`
4. `models/classifier.py`
5. `models/ocr.py`
6. `models/keypoints.py`
7. `processing/video.py`
8. `processing/court.py`
9. `processing/paths.py`
10. `processing/shots.py`
11. `utils/validators.py`
12. `visualization/annotators.py`
13. `visualization/court_renderer.py`
14. `pipeline.py`
15. `cli.py`

### Step 5: Test with sample video
```bash
python -m hoopmania analyze data/videos/sample.mp4 --output-dir outputs/
```

---

## ⚠️ RTX 4090 Optimizations

1. **Use bfloat16 for SAM2** - Already in notebook, ensure it's preserved
2. **Batch inference where possible** - Number OCR can batch crops
3. **CUDA memory management** - Clear cache between major phases
4. **Use inference-gpu** not inference - Enables GPU acceleration for Roboflow models

---

## 📋 Key Dependencies to Pin

| Package | Version | Reason |
|---------|---------|--------|
| cachetools | 5.5.0 | MRUCache removed in 6.0 |
| supervision | 0.27.0 | API compatibility |
| torch | 2.0+ | bfloat16 support |

---

## 🎮 Sample CLI Usage

```bash
# Full analysis
hoopmania analyze game.mp4 --team1 "Boston Celtics" --team2 "New York Knicks"

# Quick mode (skip OCR)
hoopmania analyze game.mp4 --skip-ocr

# Just detection + tracking (fastest)
hoopmania analyze game.mp4 --skip-ocr --skip-court-map

# Setup/download models
hoopmania setup
```

---

## 📝 Notes for Claude Code

1. **Start with minimal working version** - Get detection → tracking → export working first
2. **Add features incrementally** - Team classification, then OCR, then court mapping
3. **Test each module independently** - Write simple test scripts
4. **Handle errors gracefully** - Video codec issues, model download failures, etc.
5. **Use tqdm for all loops** - User feedback during long processing
6. **Log intermediate results** - Save frame-by-frame data for debugging
