from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Модель (предпочитаем engine; если нет — fallback на pt)
MODEL_ENGINE = BASE_DIR / "models" / "best.engine"
MODEL_PT     = BASE_DIR / "models" / "best.pt"

CONF = 0.4

# Камера
CAM_INDEX = 0
CAM_W = 640
CAM_H = 480

# Стабилизация бокса
STABLE_SECONDS_REQUIRED = 3.0
MAX_CENTER_SHIFT_PX = 8
MAX_SIZE_SHIFT_PX   = 12

# Геометрия/маска внутри ROI
MIN_CONTOUR_AREA = 300

# Сохранения
DATA_DIR = BASE_DIR / "data"
DETECTIONS_DIR = DATA_DIR / "detections"
LOGS_DIR       = DATA_DIR / "logs"

SAVE_JPG  = True
SAVE_JSON = True
SAVE_MASK = True   # full-frame binary mask png
DEBUG_SHOW_RAW = False
