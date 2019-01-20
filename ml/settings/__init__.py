import os
from pathlib import Path

BASE_DIR = Path(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
DATA_DIR = BASE_DIR / "data"
AUDIO_EVENT_DATASET_PATH = DATA_DIR / "AudioEventDataset"


# If local.py is available, Load those local settings that may override the defaults above.
# This is be useful because each computer may have the files stored in different locations.
try:
    from ml.settings.local import *
except ImportError:
    print(
        "ml/settings/local.py not found. You can make one by copying"
        " ml/settings/local.py.example to ml/settings/local.py."
    )
