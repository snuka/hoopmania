"""Configuration for HoopMania."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
FONTS_DIR = DATA_DIR / "fonts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAM2_DIR = PROJECT_ROOT / "segment-anything-2-real-time"

# SAM2 Configuration
SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

# Model IDs (from Roboflow)
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
KEYPOINT_DETECTION_MODEL_ID = "basketball-court-detection-2/14"
NUMBER_RECOGNITION_MODEL_ID = "basketball-jersey-numbers-ocr/3"

# Detection Confidence Thresholds
PLAYER_DETECTION_CONFIDENCE = 0.4
PLAYER_DETECTION_IOU_THRESHOLD = 0.9
KEYPOINT_DETECTION_CONFIDENCE = 0.3
KEYPOINT_ANCHOR_CONFIDENCE = 0.5

# Class IDs from the detection model
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
    "rim": 9,
}

# Player-related class IDs (for filtering)
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block
NUMBER_CLASS_ID = 2
BALL_IN_BASKET_CLASS_ID = 1
JUMP_SHOT_CLASS_ID = 5
LAYUP_DUNK_CLASS_ID = 6

# Team Colors (hex)
TEAM_COLORS = {
    "New York Knicks": "#006BB6",
    "Boston Celtics": "#007A33",
    "Los Angeles Lakers": "#552583",
    "Golden State Warriors": "#1D428A",
    "Miami Heat": "#98002E",
    "Chicago Bulls": "#CE1141",
    "Brooklyn Nets": "#000000",
    "Philadelphia 76ers": "#006BB6",
    "Milwaukee Bucks": "#00471B",
    "Denver Nuggets": "#0E2240",
    "Phoenix Suns": "#1D1160",
    "Dallas Mavericks": "#00538C",
    "Memphis Grizzlies": "#5D76A9",
    "Cleveland Cavaliers": "#860038",
    "Sacramento Kings": "#5A2D81",
    "Orlando Magic": "#0077C0",
    "Indiana Pacers": "#002D62",
    "Atlanta Hawks": "#E03A3E",
    "Toronto Raptors": "#CE1141",
    "Charlotte Hornets": "#1D1160",
    "Detroit Pistons": "#C8102E",
    "Oklahoma City Thunder": "#007AC1",
    "Minnesota Timberwolves": "#0C2340",
    "Portland Trail Blazers": "#E03A3E",
    "New Orleans Pelicans": "#0C2340",
    "Utah Jazz": "#002B5C",
    "San Antonio Spurs": "#C4CED4",
    "Houston Rockets": "#CE1141",
    "Washington Wizards": "#002B5C",
    "Los Angeles Clippers": "#C8102E",
}

# Team Rosters (jersey number -> player name)
TEAM_ROSTERS = {
    "New York Knicks": {
        "55": "Hukporti",
        "1": "Payne",
        "0": "Wright",
        "11": "Brunson",
        "3": "Hart",
        "32": "Towns",
        "44": "Shamet",
        "25": "Bridges",
        "2": "McBride",
        "23": "Robinson",
        "8": "Anunoby",
        "4": "Dadiet",
        "5": "Achiuwa",
        "13": "Kolek",
    },
    "Boston Celtics": {
        "42": "Horford",
        "55": "Scheierman",
        "9": "White",
        "20": "Davison",
        "7": "Brown",
        "0": "Tatum",
        "27": "Walsh",
        "4": "Holiday",
        "8": "Porzingis",
        "40": "Kornet",
        "88": "Queta",
        "11": "Pritchard",
        "30": "Hauser",
        "12": "Craig",
        "26": "Tillman",
    },
}

# Visualization Colors
ANNOTATION_COLORS = [
    "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
    "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
]

# Processing Settings
OCR_FRAME_INTERVAL = 5  # Run OCR every N frames
CONSECUTIVE_VALIDATION_THRESHOLD = 3  # Number of consecutive reads to validate a jersey number
PATH_SMOOTHING_WINDOW = 9
PATH_SMOOTHING_POLY = 2

# Shot Event Detection Settings
SHOT_RESET_TIME_SECONDS = 1.7
SHOT_MIN_FRAMES_BETWEEN_STARTS = 0.5  # seconds
SHOT_COOLDOWN_AFTER_MADE = 0.5  # seconds
