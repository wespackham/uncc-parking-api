import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
LGB_MODELS_DIR = BASE_DIR / "models_lgb_3h"
LGB_MODELS_V2_DIR = BASE_DIR / "models_lgb_24h"
LGB_MODELS_V3_DIR = BASE_DIR / "models_lgb_v3"
LGB_MODELS_V4_DIR = BASE_DIR / "models_lgb_v4"
LGB_MODELS_24H_V2_DIR = BASE_DIR / "models_lgb_24h_v2"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
PREDICTIONS_BUFFER_FILE = LOGS_DIR / "pending_predictions.jsonl"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
# Optional prefix so messages from different hosts can be told apart, e.g. "droplet".
DISCORD_LABEL = os.environ.get("DISCORD_LABEL", "")
API_KEY = os.environ.get("API_KEY", "")
# Lots whose live feed is known-broken: predictions are not written for them (comma-separated).
SUPPRESSED_LOTS = {lot.strip() for lot in os.environ.get("SUPPRESSED_LOTS", "").split(",") if lot.strip()}

TABLE_PARKING_DATA = "parking_data"
TABLE_PREDICTIONS = "parking_predictions"

LOTS = ["CRI", "ED1", "UDL", "UDU", "WEST", "CD FS", "CD VS", "ED2/3", "NORTH", "SOUTH"]

LAT = 35.3076
LON = -80.7291


def discord_labeled(message: str) -> str:
    return f"**[{DISCORD_LABEL}]** {message}" if DISCORD_LABEL else message


def safe_name(lot: str) -> str:
    return lot.replace(" ", "_").replace("/", "_")
