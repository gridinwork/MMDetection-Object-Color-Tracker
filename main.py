"""MMDetection Object & Color Vision Studio."""

from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import setup_logging
from utils.paths import ensure_dirs


def main() -> int:
    faulthandler.enable()
    logger = setup_logging()
    logger.info("application start")
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow
    from app.theme import THEME

    app = QApplication(sys.argv)
    app.setApplicationName("MMDetection Object & Color Vision Studio")
    app.setStyleSheet(THEME)
    window = MainWindow()
    window.show()
    code = app.exec()
    logger.info("application exit %s", code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
