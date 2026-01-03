# HoopMania Pivot: Manual Player Selection + Tracking

## Overview

**Problem**: Detection and OCR are unreliable (~60% accuracy). Users spend more time correcting AI errors than getting value.

**Solution**: Let users manually identify players they want to track. They already know who's who - leverage that knowledge.

**Target User**: Parent or coach filming their kid's/team's game who wants player-specific stats.

---

## Key Insight: Hybrid Detection + Manual ID

**Don't remove detection entirely - use it for SELECTION, not IDENTIFICATION.**

```
Current (All Automated - Fragile):
Video → Detect → Classify Team → OCR Number → Track → Stats
        ↑ fails   ↑ can flip     ↑ 60% acc

Pivot (Hybrid - Robust):
Video → Detect Boxes → User Clicks Box → User Enters Name/# → Track → Stats
        ↑ works well   ↑ 100% accurate   ↑ 100% accurate
```

**Why this is better than raw clicks:**
- Detection gives precise bounding boxes (better tracking initialization)
- Easier to click a highlighted box than find the exact player boundary
- Shows "12 players detected - select up to 5" (clear affordance)
- Still no OCR, no team classification - user provides identity

---

## Architecture

### Components

| Component | Status | Purpose |
|-----------|--------|---------|
| PlayerDetector | **KEEP** | Detect boxes for selection UI (not for ID) |
| JerseyOCR | **REMOVE** | User provides jersey numbers |
| TeamClassifier | **REMOVE** | User assigns teams |
| SAM2Tracker | **KEEP** | Core tracking engine |
| CourtKeypointDetector | **KEEP** | Court position mapping |
| CourtMapper | **KEEP** | Transform to court coordinates |
| StatsEngine | **NEW** | Calculate player statistics |
| SubstitutionDetector | **NEW** | Detect when players leave court |
| PDFExporter | **NEW** | Generate stats reports |

### Data Model

```python
@dataclass
class Player:
    """A player in the roster (persists across substitutions)."""
    id: str                      # UUID
    name: str                    # "Marcus"
    number: str                  # "23"
    team: str                    # "My Team" or "Celtics"
    color: Tuple[int, int, int]  # RGB for visualization

@dataclass
class Roster:
    """All players who might appear in the game."""
    my_team: List[Player]        # Up to 12-15 players
    opponent_team: List[Player]  # Optional, for matchup analysis

@dataclass
class TrackingAnchor:
    """A point where user identified a player on court."""
    player_id: str
    frame_idx: int
    bbox: Tuple[float, float, float, float]

@dataclass
class Segment:
    """A continuous tracking segment (between substitutions)."""
    start_frame: int
    end_frame: int
    active_players: List[str]    # Player IDs currently on court
    anchors: List[TrackingAnchor]

@dataclass
class GameSession:
    """Complete game analysis."""
    video_path: Path
    roster: Roster
    segments: List[Segment]
    fps: float
    court_mapping_available: bool
```

---

## User Flow (Revised)

### Flow 1: Quick Mode (Single Segment)
For short clips or when user doesn't care about substitutions.

```
1. Upload Video
2. Auto-detect or scrub to reference frame
3. System shows detected player boxes (numbered)
4. User clicks boxes for players to track (max 5)
   └── For each: Enter Name, Number, Team
5. Click "Track Players"
6. View results: Video + Stats + Export
```

### Flow 2: Game Mode (Multiple Segments with Roster)
For full games with substitutions.

```
1. Upload Video
2. Define Roster (optional, can add during tracking)
   ├── My Team: Marcus #23, Jordan #11, Tyler #5, ...
   └── Opponent: (optional)
3. Select reference frame for first segment
4. Click on 5 players currently on court
   └── Assign from roster dropdown (or create new)
5. Click "Start Tracking"
6. When substitution detected:
   ├── System pauses: "Marcus #23 left at 4:32"
   ├── User confirms substitution
   ├── User clicks replacement player
   └── User assigns from roster: "Tyler #5 entering"
7. Tracking continues
8. View results: Per-player stats aggregated across segments
```

### Substitution UX Detail

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠️ SUBSTITUTION DETECTED                                      │
│                                                                 │
│  Marcus Johnson (#23) appears to have left the court at 4:32   │
│                                                                 │
│  What happened?                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ○ Substitution - select replacement player              │   │
│  │ ○ Tracking lost - re-anchor Marcus at current frame     │   │
│  │ ○ End of quarter - pause tracking here                  │   │
│  │ ○ Ignore - Marcus will return shortly                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Show Frame at 4:32]  [Continue Without Action]               │
└─────────────────────────────────────────────────────────────────┘
```

This distinguishes between:
- **Substitution**: Player went to bench, new player enters
- **Tracking Lost**: SAM2 lost the mask, need to re-anchor same player
- **End of Quarter**: Natural break, will resume later
- **Temporary Occlusion**: Player will reappear, just wait

---

## Stats Engine (Expanded)

### Core Stats (From Position Data)

| Stat | Description | Units | Requires Court Mapping? |
|------|-------------|-------|------------------------|
| Play Time | Time visible on court | minutes | No |
| Distance | Total ground covered | feet/miles | Yes (relative without) |
| Avg Speed | Mean movement speed | mph | Yes (relative without) |
| Max Speed | Peak sprint speed | mph | Yes (relative without) |
| Paint Time | Time in the paint | minutes | Yes |
| Perimeter Time | Time outside paint | minutes | Yes |
| 3PT Zone Time | Time beyond arc | minutes | Yes |
| Corner Time | Time in corners | minutes | Yes |

### Hustle Stats (From Movement Patterns)

| Stat | Description | Calculation |
|------|-------------|-------------|
| Sprint Count | Times player accelerated hard | Frames where accel > threshold |
| Direction Changes | Quick cuts/pivots | Frames where heading changes >90° |
| Stationary Time | Time standing still | Frames where speed < 0.5 mph |
| Active Time % | Moving vs standing | (Play Time - Stationary) / Play Time |

### Heatmap Zones

```
┌─────────────────────────────────────────────┐
│                                             │
│  ┌───────┐             ┌───────┐           │
│  │Corner │             │Corner │           │
│  │ Left  │             │ Right │           │
│  └───────┘             └───────┘           │
│         ╲             ╱                     │
│          ╲ 3PT ARC  ╱                      │
│           ╲       ╱                         │
│  ┌─────────────────────────────┐           │
│  │         MID-RANGE           │           │
│  └─────────────────────────────┘           │
│        ┌─────────────────┐                 │
│        │      PAINT      │                 │
│        │    (THE KEY)    │                 │
│        └─────────────────┘                 │
│              [BASKET]                       │
└─────────────────────────────────────────────┘
```

Time breakdown per zone gives coaching insights:
- "Marcus spends 60% of time in paint" → Post player
- "Jordan mostly in corners" → 3PT specialist
- "Tyler high perimeter time" → Ball handler

### Fallback: Relative Stats (No Court Mapping)

If court keypoint detection fails:
- Distance in pixels (still useful for comparison)
- Speed in pixels/second
- Heatmap in frame coordinates
- Zone stats unavailable
- Clear disclaimer: "Court mapping unavailable - stats are relative"

---

## Handling Edge Cases

### 1. Player Not in Reference Frame
**Problem**: User wants to track a player who isn't visible in the auto-selected frame.

**Solution**: Allow adding players at ANY frame, not just one reference.
```
[Add Player at Different Frame]
     ↓
[Scrub to frame where player is visible]
     ↓
[Click + identify player]
     ↓
[System tracks bidirectionally from that anchor point]
```

### 2. Camera Cuts / Replays
**Problem**: Broadcast footage has cuts to crowd, replays, etc.

**Solution**: Detect sudden frame changes and handle gracefully.
- Large pixel difference between frames → likely camera cut
- Option: "Mark this section as replay/break - skip tracking"
- SAM2 naturally handles brief occlusions; longer breaks need user input

### 3. Player Returns After Substitution
**Problem**: Player gets subbed out, then subbed back in later.

**Solution**: Roster concept handles this.
- Player identity persists in roster
- When they return: "Marcus #23 returning at 12:45 - click to track"
- Stats aggregate across all their segments

### 4. Tracking Confidence Drop
**Problem**: SAM2 mask becomes unreliable but player is still on court.

**Solution**: Expose tracking confidence to user.
- Show confidence indicator per player
- When confidence drops: "Tracking for Marcus is uncertain - verify or re-anchor"
- Allow mid-video re-anchoring without creating new segment

---

## Visualization Options

### Option A: Focused View (Default)
Only selected players are highlighted. Others are dimmed or invisible.
- Clean, focuses attention
- Good for parents tracking one kid

### Option B: Context View
Selected players have full labels. Other detected players have generic boxes.
- Shows game context
- Good for coaches analyzing positioning

### Option C: Minimap Mode
Full video unchanged, but corner shows court diagram with player dots.
- Non-intrusive
- Good for tactical analysis

User can toggle between these.

---

## Implementation Phases (Revised)

### Phase 1: Core Selection + Tracking (MVP)

**1A: Detection-Assisted Selection UI**
- Upload video
- Auto-select or manual scrub to reference frame
- Run player detection → show numbered boxes
- Click boxes to select players
- Enter name/number for each
- Simple team assignment (My Team / Opponent)

**1B: SAM2 Tracking Integration**
- Initialize SAM2 with selected player bboxes
- Bidirectional tracking from reference frame
- Basic progress display

**1C: Basic Output**
- Annotated video with player labels
- Simple stats display: play time, distance (relative)

### Phase 2: Stats Engine

**2A: Court Position Extraction**
- Integrate court keypoint detection
- Map player positions to court coordinates
- Handle mapping failures gracefully

**2B: Full Stats Calculation**
- All core stats (distance, speed, zones)
- Hustle metrics
- Per-player heatmaps

**2C: Stats UI**
- Dashboard view with per-player cards
- Heatmap visualization
- Comparison view (if multiple players)

### Phase 3: Substitution Handling

**3A: Detection Logic**
- Track when player mask disappears
- Distinguish: substitution vs tracking lost vs occlusion
- Detect segment boundaries (quarter breaks)

**3B: Roster System**
- Pre-define player roster
- Assign players from roster during tracking
- Persist identity across substitutions

**3C: Re-tagging UI**
- Pause and notify on substitution
- Show options: sub / re-anchor / break / ignore
- Smooth continuation of tracking

### Phase 4: Export

**4A: PDF Report**
- Per-player stats summary
- Heatmaps embedded
- Professional formatting

**4B: Video Export Options**
- Full annotated video
- Highlight clips (mark key moments manually for now)
- Per-player "follow cam" clip

### Phase 5: Polish & Future

**5A: Multi-video support**
- Aggregate stats across multiple video files
- Handle full games recorded in parts

**5B: Ball tracking (future)**
- User clicks ball in addition to players
- Enables possession, shot detection

---

## Files to Reuse from Previous Branch

These were built in `feature/progress-tracking-gpu` and can be cherry-picked:

| File | Purpose | Reuse? |
|------|---------|--------|
| `utils/progress.py` | Progress tracking with GPU/ETA | ✅ Yes |
| `utils/frame_selector.py` | Reference frame selection + detection assist | ✅ Yes (core of new UI) |
| `pipeline.py` (parts) | Bidirectional tracking logic | ✅ Partial |
| `models/detector.py` | Player detection | ✅ Yes (for selection UI) |
| `models/sam2_tracker.py` | SAM2 tracking | ✅ Yes |

---

## File Structure (Final)

```
hoopmania/
├── app.py                          # Gradio UI (simplified, focused)
├── PIVOT_PLAN.md                   # This document
├── hoopmania/
│   ├── config.py                   # Simplified config
│   ├── pipeline.py                 # ManualTrackingPipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py             # KEEP - for selection UI
│   │   ├── sam2_tracker.py         # KEEP - core tracking
│   │   └── court_keypoints.py      # KEEP - for stats
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── session.py              # NEW - GameSession, Segment, Roster
│   │   └── substitution.py         # NEW - SubstitutionDetector
│   │
│   ├── stats/
│   │   ├── __init__.py
│   │   ├── engine.py               # NEW - StatsEngine
│   │   ├── metrics.py              # NEW - Individual metric functions
│   │   ├── zones.py                # NEW - Court zone definitions
│   │   └── heatmap.py              # NEW - Heatmap generation
│   │
│   ├── processing/
│   │   ├── court_mapper.py         # KEEP
│   │   └── video.py                # KEEP
│   │
│   ├── visualization/
│   │   ├── annotator.py            # SIMPLIFY - just labels
│   │   ├── court_renderer.py       # KEEP + enhance for heatmaps
│   │   └── dashboard.py            # NEW - stats dashboard components
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   ├── video.py                # Video export
│   │   └── pdf.py                  # PDF report generation
│   │
│   └── utils/
│       ├── progress.py             # REUSE from previous branch
│       └── frame_selector.py       # REUSE from previous branch
```

---

## UI Wireframe (Revised)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  HOOPMANIA - Track Your Players                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: Upload Video                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  [Choose File]  sample_game.mp4                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STEP 2: Select Reference Frame                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │              [VIDEO FRAME WITH DETECTED PLAYER BOXES]          │   │
│  │                                                                 │   │
│  │     ┌──[1]──┐   ┌──[2]──┐   ┌──[3]──┐                         │   │
│  │     │      │   │      │   │      │   ... (detected players)  │   │
│  │     └───────┘   └───────┘   └───────┘                         │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  [Auto-Select Best Frame]    ◄━━━━━━━━━━●━━━━━━━━━━►    Frame: 45/1501 │
│                                                                         │
│  STEP 3: Select Players to Track (click boxes above)                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ✓ Player 1: [Marcus_____] #[23] [My Team     ▼] 🟢 [Remove]  │   │
│  │  ✓ Player 2: [Jordan_____] #[11] [My Team     ▼] 🔵 [Remove]  │   │
│  │  ○ Player 3: Click a box above to add...                       │   │
│  │  ○ Player 4:                                                    │   │
│  │  ○ Player 5:                                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [▶ Start Tracking]                                                     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Progress: 67.3% | ETA: 02:15 | GPU: 94% | Tracking 2 players          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RESULTS  [Video] [Stats] [Export PDF]                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐                              │   │
│  │  │ MARCUS #23  │  │ JORDAN #11  │                              │   │
│  │  │ Play: 8:34  │  │ Play: 8:34  │                              │   │
│  │  │ Dist: 0.6mi │  │ Dist: 0.8mi │                              │   │
│  │  │ Spd: 4.2mph │  │ Spd: 5.1mph │                              │   │
│  │  │ [Heatmap]   │  │ [Heatmap]   │                              │   │
│  │  └─────────────┘  └─────────────┘                              │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [Download Video]  [Download PDF Report]                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Makes This Plan Better

| Aspect | Original Plan | Improved Plan |
|--------|---------------|---------------|
| Player selection | Raw clicks (imprecise) | Detection-assisted boxes (precise) |
| Substitution handling | Reactive only | Proactive options (sub/lost/break/ignore) |
| Player identity | Per-segment only | Roster persists across segments |
| Court mapping failure | Not addressed | Graceful fallback with relative stats |
| Stats depth | Basic 6 stats | 12+ stats including hustle metrics |
| Multi-anchor | One reference frame | Add players at any frame |
| Code reuse | Start fresh | Cherry-pick from previous branch |
| Tracking issues | Generic "disappeared" | Distinguish lost vs substituted |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Player selection | <30 sec for 5 players |
| Tracking accuracy | >95% (SAM2 baseline) |
| Stats accuracy | ±5% vs manual measurement |
| Processing time | <2x video duration |
| Substitution detection | >90% recall |
| Court mapping success | >80% of frames |
| PDF generation | <5 seconds |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Court mapping fails frequently | Relative stats fallback, manual corner marking (v2) |
| SAM2 tracking drifts over long videos | Re-anchor capability, shorter segments |
| Substitution detection false positives | User confirmation required, multiple options |
| User finds selection tedious | Detection-assisted selection reduces clicks |
| Stats seem inaccurate | Validate against manual stopwatch, show confidence |

---

## Next Steps

1. **Approve this plan**
2. **Cherry-pick reusable code** from `feature/progress-tracking-gpu`:
   - `utils/progress.py`
   - `utils/frame_selector.py`
   - Parts of `pipeline.py` (bidirectional tracking)
3. **Start Phase 1A**: Detection-assisted selection UI
4. **Iterate** based on testing

Ready to begin implementation?
