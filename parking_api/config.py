import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_V2_DIR = BASE_DIR / "models_v2"
LGB_MODELS_DIR = BASE_DIR / "models_lgb"
DATA_DIR = BASE_DIR / "data"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
API_KEY = os.environ.get("API_KEY", "")

TABLE_PARKING_DATA = "parking_data"
TABLE_PREDICTIONS = "parking_predictions"

LOTS = ["CRI", "ED1", "UDL", "UDU", "WEST", "CD FS", "CD VS", "ED2/3", "NORTH", "SOUTH"]

LAT = 35.3076
LON = -80.7291


def safe_name(lot: str) -> str:
    return lot.replace(" ", "_").replace("/", "_")
