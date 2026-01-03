# HoopMania Pivot: Manual Player Selection + Tracking

## Overview

**Problem**: Detection and OCR are unreliable (~60% accuracy). Users spend more time correcting AI errors than getting value.

**Solution**: Let users manually identify players they want to track. They already know who's who - leverage that knowledge.

**Target User**: Parent or coach filming their kid's/team's game who wants player-specific stats.

---

## Architecture Comparison

### Before (Complex, Fragile)
```
Video → PlayerDetector → TeamClassifier → JerseyOCR → SAM2 → Stats
         (unreliable)     (can flip)      (60% acc)   (good!)
```

### After (Simple, Reliable)
```
Video → User Click + Input → SAM2 → CourtMapper → StatsEngine → Export
        (100% accurate)      (good!)  (reuse)      (new)        (new)
```

**Removed**: PlayerDetector (for ID), JerseyOCR, TeamClassifier
**Kept**: SAM2Tracker, CourtMapper, CourtKeypointDetector
**New**: StatsEngine, SubstitutionDetector, PDFExporter, HighlightClipper (future)

---

## User Flow

```
1. Upload Video
2. Select Reference Frame (auto or manual scrub)
3. Click on Players to Track (max 5)
   └── For each click:
       ├── Enter Name (e.g., "Marcus")
       ├── Enter Number (e.g., "23")
       └── Select Team (dropdown or "My Team" / "Opponent")
4. Click "Start Tracking"
5. [System processes video with SAM2]
6. [System detects substitutions, prompts re-tag if needed]
7. View Results:
   ├── Annotated Video (only selected players labeled)
   ├── Stats Dashboard (per player)
   └── Export Options (video file, PDF report)
```

---

## Phase 1: Core Manual Selection + Tracking

### 1.1 Simplified Data Model

```python
@dataclass
class TrackedPlayer:
    """A player selected by the user for tracking."""
    name: str                    # "Marcus"
    number: str                  # "23"
    team: str                    # "My Team" or team name
    color: Tuple[int, int, int]  # RGB for visualization
    initial_bbox: Tuple[float, float, float, float]  # From user click
    tracker_id: int              # Assigned by SAM2

@dataclass
class TrackingSession:
    """A tracking session (one per video or per substitution segment)."""
    video_path: Path
    start_frame: int
    end_frame: int  # -1 means end of video
    players: List[TrackedPlayer]

@dataclass
class AnalysisConfig:
    """Simplified config for manual tracking."""
    output_dir: Path
    generate_video: bool = True
    generate_stats: bool = True
    generate_pdf: bool = True
```

### 1.2 UI Components (app.py)

```
┌─────────────────────────────────────────────────────────────────┐
│  HoopMania - Track Your Players                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Upload Video]                                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │            Reference Frame Preview                      │   │
│  │       (click on players to select them)                 │   │
│  │                                                         │   │
│  │    [Auto-Select Best Frame]  [◄] ━━━●━━━━━ [►]         │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Selected Players:                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ #1  [Marcus    ] [23  ] [My Team ▼] [🎨] [✕ Remove]     │  │
│  │ #2  [Jordan    ] [11  ] [My Team ▼] [🎨] [✕ Remove]     │  │
│  │ #3  [Click on a player to add...]                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  [Start Tracking]                                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Progress: 45.2% | ETA: 03:15 | GPU: 92%                       │
├─────────────────────────────────────────────────────────────────┤
│  Results:                                                       │
│  [Video] [Stats] [Export PDF]                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Pipeline (pipeline.py)

```python
class ManualTrackingPipeline:
    """Simplified pipeline for manual player tracking."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._tracker = SAM2Tracker()
        self._court_mapper = CourtMapper()
        self._keypoint_detector = CourtKeypointDetector()

    def track(
        self,
        video_path: Path,
        players: List[TrackedPlayer],
        reference_frame: int = 0,
        progress_callback = None,
    ) -> TrackingResult:
        """Track selected players through video."""
        # 1. Initialize SAM2 with user-selected player boxes
        # 2. Run tracking (forward + backward from reference)
        # 3. Map court positions
        # 4. Detect substitutions (player disappears)
        # 5. Return tracking data
```

---

## Phase 2: Stats Generation

### 2.1 Stats to Calculate

| Stat | Description | Calculation |
|------|-------------|-------------|
| **Play Time** | Time player is on court | Frames visible / FPS |
| **Distance** | Total distance traveled | Sum of court position deltas |
| **Avg Speed** | Average movement speed | Distance / Play Time |
| **Max Speed** | Peak speed achieved | Max of frame-to-frame speeds |
| **Paint Time** | Time in the paint | Frames where court_y > threshold |
| **Perimeter Time** | Time on perimeter | Frames outside paint |
| **Heatmap** | Position density | 2D histogram of court positions |

### 2.2 StatsEngine Class

```python
@dataclass
class PlayerStats:
    player: TrackedPlayer
    play_time_seconds: float
    distance_feet: float
    avg_speed_mph: float
    max_speed_mph: float
    paint_time_seconds: float
    perimeter_time_seconds: float
    heatmap: np.ndarray  # 2D density array
    positions: List[Tuple[float, float]]  # All court positions

class StatsEngine:
    """Calculate player statistics from tracking data."""

    COURT_LENGTH_FEET = 94
    COURT_WIDTH_FEET = 50
    PAINT_Y_THRESHOLD = 0.19  # ~19 feet from baseline

    def calculate(
        self,
        player: TrackedPlayer,
        court_positions: List[np.ndarray],
        fps: float,
    ) -> PlayerStats:
        """Calculate all stats for a player."""
```

---

## Phase 3: Substitution Detection

### 3.1 Detection Logic

A substitution is likely when:
1. Tracked player's bounding box disappears for >10 seconds
2. Player position is near sideline/bench area before disappearing
3. Tracking confidence drops significantly

### 3.2 User Notification Flow

```
[System detects Player #23 disappeared at 4:32]
     ↓
[Notification]: "Marcus (#23) may have been substituted at 4:32.
                 Would you like to tag a replacement?"
     ↓
[User clicks new player entering the game]
     ↓
[System]: "Who is replacing Marcus?"
     ↓
[User enters]: "Tyler, #5"
     ↓
[System continues tracking Tyler #5 for rest of video]
```

### 3.3 Implementation

```python
class SubstitutionDetector:
    """Detect when tracked players leave the court."""

    def __init__(self, disappear_threshold_seconds: float = 10.0):
        self.threshold = disappear_threshold_seconds
        self._last_seen: Dict[int, int] = {}  # tracker_id -> last frame

    def update(self, frame_idx: int, detections: sv.Detections) -> List[int]:
        """Update with current frame, return list of substituted tracker_ids."""

    def get_substitution_events(self) -> List[SubstitutionEvent]:
        """Get all detected substitution events with timestamps."""
```

---

## Phase 4: Export

### 4.1 Video Export
- Annotated video with only selected players labeled
- Player name + number displayed above each player
- Team color for bounding box/mask
- Court minimap in corner (optional)

### 4.2 PDF Report

```
┌─────────────────────────────────────────────────────────────────┐
│                    PLAYER STATS REPORT                          │
│                    Game: vs Opponents                           │
│                    Date: 2024-01-03                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MARCUS JOHNSON (#23) - My Team                                │
│  ───────────────────────────────────────────────────────────── │
│                                                                 │
│  Play Time:     18:34          Distance:    1.2 miles          │
│  Avg Speed:     4.2 mph        Max Speed:   14.8 mph           │
│  Paint Time:    4:12           Perimeter:   14:22              │
│                                                                 │
│  ┌─────────────────────────────┐                               │
│  │                             │                               │
│  │     [COURT HEATMAP]        │                               │
│  │                             │                               │
│  └─────────────────────────────┘                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  JORDAN SMITH (#11) - My Team                                  │
│  ...                                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Future Enhancements

### 5.1 Ball Tracking
- User clicks ball in addition to players
- Enables: possession time, shot detection, assist tracking

### 5.2 Highlight Clip Generation
- When ball + player proximity detected, mark as "involvement"
- Auto-generate clips of player's key moments
- Export as separate video or timestamps

### 5.3 Multi-Segment Tracking
- Handle full games with multiple quarters
- Automatic break detection (no movement for >30 seconds)
- Per-quarter stats breakdown

---

## File Structure (Pivot Branch)

```
hoopmania/
├── app.py                      # Simplified Gradio UI
├── hoopmania/
│   ├── config.py               # Simplified config
│   ├── pipeline.py             # ManualTrackingPipeline (simplified)
│   ├── models/
│   │   ├── sam2_tracker.py     # Keep as-is
│   │   └── court_keypoints.py  # Keep as-is
│   ├── processing/
│   │   ├── court_mapper.py     # Keep as-is
│   │   ├── stats_engine.py     # NEW: Calculate player stats
│   │   └── substitution.py     # NEW: Detect substitutions
│   ├── visualization/
│   │   ├── annotator.py        # Simplified for manual labels
│   │   ├── court_renderer.py   # Keep as-is
│   │   └── heatmap.py          # NEW: Generate heatmaps
│   ├── export/
│   │   ├── video.py            # Video export with annotations
│   │   └── pdf_report.py       # NEW: PDF stats report
│   └── utils/
│       ├── progress.py         # Reuse from previous branch
│       └── frame_selector.py   # Reuse from previous branch
```

---

## Implementation Order

1. **Phase 1A**: UI for player selection (click + name/number input)
2. **Phase 1B**: Connect to SAM2 tracking with manual prompts
3. **Phase 1C**: Basic annotated video output
4. **Phase 2A**: Court position extraction (reuse existing)
5. **Phase 2B**: StatsEngine implementation
6. **Phase 2C**: Stats display in UI
7. **Phase 3A**: Substitution detection
8. **Phase 3B**: Re-tagging UI flow
9. **Phase 4A**: PDF report generation
10. **Phase 4B**: Export UI (download buttons)

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Player selection time | <30 seconds for 5 players |
| Tracking accuracy | >95% (SAM2 baseline) |
| Stats accuracy | ±5% vs manual measurement |
| Processing time | <2x video duration |
| User satisfaction | "This is exactly what I needed" |

---

## What We're NOT Building

- Automatic player detection (that's the whole point)
- Jersey OCR (user provides this)
- Team classification (user assigns team)
- Play-by-play detection (future, needs ball)
- Referee/coach filtering (only track what user clicks)

---

## Next Steps

1. Review and approve this plan
2. Start with Phase 1A: Click-to-select UI
3. Iterate based on testing

Ready to begin implementation?
