"""Download, delete and verify local MMDetection checkpoints."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from inference.model_registry import ModelRegistry
from utils.downloader import DownloadError, download_file


class _TaskThread(QThread):
    progress = Signal(int, int)
    ok = Signal(str)
    failed = Signal(str)

    def __init__(self, kind: str, spec, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.spec = spec

    def run(self) -> None:
        try:
            if self.kind == "download":
                if not self.spec.checkpoint_url:
                    raise DownloadError("This model has no download URL. Place the checkpoint manually.")
                download_file(
                    self.spec.checkpoint_url,
                    self.spec.checkpoint_path,
                    progress=lambda done, total: self.progress.emit(done, total),
                )
                self._verify()
            else:
                self._verify()
            self.ok.emit(self.spec.display_name)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _verify(self) -> None:
        import torch

        blob = torch.load(str(self.spec.checkpoint_path), map_location="cpu")
        if not isinstance(blob, dict):
            raise RuntimeError("Checkpoint is not a PyTorch state dictionary")


class ModelManagerDialog(QDialog):
    def __init__(self, registry: ModelRegistry, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("Model Manager")
        self.resize(760, 460)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model", "Task", "Status", "Checkpoint"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._show_description)
        self.description = QLabel("")
        self.description.setWordWrap(True)
        download = QPushButton("DOWNLOAD")
        delete = QPushButton("DELETE")
        verify = QPushButton("VERIFY")
        close = QPushButton("CLOSE")
        download.clicked.connect(self._download)
        delete.clicked.connect(self._delete)
        verify.clicked.connect(self._verify)
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(download)
        buttons.addWidget(delete)
        buttons.addWidget(verify)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.description)
        layout.addLayout(buttons)
        self._thread: _TaskThread | None = None
        self._progress: QProgressDialog | None = None
        self.refresh()

    def refresh(self) -> None:
        self.registry.reload_custom()
        models = self.registry.all()
        self.table.setRowCount(len(models))
        for row, spec in enumerate(models):
            status = "INSTALLED" if spec.installed else "NOT INSTALLED"
            task = "Instance Segmentation" if spec.supports_masks else "Detection"
            self.table.setItem(row, 0, QTableWidgetItem(spec.display_name))
            self.table.setItem(row, 1, QTableWidgetItem(task))
            self.table.setItem(row, 2, QTableWidgetItem(status))
            self.table.setItem(row, 3, QTableWidgetItem(spec.checkpoint_file))
            self.table.item(row, 0).setData(256, spec.model_id)
        self.table.resizeColumnsToContents()

    def _selected(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        model_id = self.table.item(row, 0).data(256)
        return self.registry.get(str(model_id))

    def _show_description(self) -> None:
        spec = self._selected()
        self.description.setText(spec.description if spec else "")

    def _download(self) -> None:
        spec = self._selected()
        if spec is None:
            return
        self._start("download", spec)

    def _verify(self) -> None:
        spec = self._selected()
        if spec is None:
            return
        if not spec.installed:
            QMessageBox.warning(self, "Verify", f"{spec.display_name} is NOT INSTALLED")
            return
        self._start("verify", spec)

    def _start(self, kind: str, spec) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._progress = QProgressDialog(f"{kind.capitalize()} {spec.display_name}", None, 0, 100, self)
        self._progress.setWindowTitle("Model Manager")
        self._progress.setCancelButton(None)
        self._progress.setMinimumDuration(0)
        self._progress.setValue(0)
        self._thread = _TaskThread(kind, spec, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.ok.connect(self._on_ok)
        self._thread.failed.connect(self._on_fail)
        self._thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        if self._progress is None:
            return
        if total <= 0:
            self._progress.setRange(0, 0)
            return
        self._progress.setRange(0, 100)
        self._progress.setValue(min(100, int(done * 100 / total)))

    def _on_ok(self, name: str) -> None:
        if self._progress is not None:
            self._progress.close()
        self.refresh()
        QMessageBox.information(self, "Model Manager", f"{name}: OK")

    def _on_fail(self, message: str) -> None:
        if self._progress is not None:
            self._progress.close()
        QMessageBox.critical(self, "Model Manager", message)

    def _delete(self) -> None:
        spec = self._selected()
        if spec is None:
            return
        path = spec.checkpoint_path
        if not path.exists():
            QMessageBox.information(self, "Model Manager", f"{spec.display_name} is already NOT INSTALLED")
            return
        answer = QMessageBox.question(self, "Delete", f"Delete {spec.display_name} checkpoint?")
        if answer != QMessageBox.Yes:
            return
        path.unlink()
        self.refresh()
