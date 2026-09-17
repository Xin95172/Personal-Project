"""Learning GUI: controls live here; audio and inference live in other modules."""
import sys
from pathlib import Path
from threading import Event
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton,
    QFileDialog, QComboBox,
)
from audio_engine import load_audio, play_audio, stop_audio
from voice_converter import VoiceConverter, ConversionCancelled

ROOT = Path(__file__).resolve().parent
AUDIO_PATH = ROOT / "p0002-7.mp3"


class ConversionWorker(QThread):
    progress = Signal(str)
    converted = Signal(object)
    failed = Signal(str)

    def __init__(self, converter, arguments, parent=None):
        super().__init__(parent)
        self.converter = converter
        self.arguments = arguments
        self.cancel = Event()

    def run(self):
        try:
            result = self.converter.convert(
                **self.arguments, progress=self.progress.emit, cancel=self.cancel)
            if not self.cancel.is_set():
                self.converted.emit(result)
        except ConversionCancelled:
            pass
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class VoiceChanger(QWidget):
    def __init__(self):
        super().__init__()
        self.converter = VoiceConverter()
        self.audio, self.sr = None, None
        self.worker = None
        self.closing = False
        self.config_path = ROOT / "models/config.json"
        self.setWindowTitle("Voice Changer — RVC learning MVP")
        self.resize(560, 460)
        layout = QVBoxLayout(self)
        self.file_label = QLabel("尚未載入音訊")
        self.load_button = QPushButton("Load Audio")
        self.backend = QComboBox()
        self.backend.addItem("AI Voice Conversion — RVC ONNX（需要模型）", "rvc")
        self.backend.addItem("DSP 比較模式 — Pitch / Speed", "dsp")
        self.config_button = QPushButton("選擇模型設定 JSON")
        self.config_label = QLabel(str(self.config_path))
        self.config_label.setWordWrap(True)
        self.pitch_label = QLabel("pitch: 0 st")
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.speed_label = QLabel("speed: 1.00x")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 150)
        self.speed_slider.setValue(100)
        self.original_button = QPushButton("Play Original")
        self.play_button = QPushButton("Play Converted")
        self.stop_button = QPushButton("Stop")
        self.status = QLabel("AI 模式需要 models/ 中的模型；可選 DSP 比較模式。")
        self.status.setWordWrap(True)
        for widget in (self.file_label, self.load_button, self.backend,
                       self.config_button, self.config_label,
                       self.pitch_label, self.pitch_slider, self.speed_label,
                       self.speed_slider, self.original_button, self.play_button,
                       self.stop_button, self.status):
            layout.addWidget(widget)
        self.pitch_slider.valueChanged.connect(self.update_pitch_label)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.load_button.clicked.connect(self.choose_audio)
        self.config_button.clicked.connect(self.choose_config)
        self.original_button.clicked.connect(self.play_original)
        self.play_button.clicked.connect(self.play_converted)
        self.stop_button.clicked.connect(self.stop)
        if AUDIO_PATH.is_file():
            self.open_audio(AUDIO_PATH)
        self.set_busy(False)

    def update_pitch_label(self, value):
        self.pitch_label.setText(f"pitch: {value} st")

    def update_speed_label(self, value):
        self.speed_label.setText(f"speed: {value / 100:.2f}x")

    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇音訊", str(ROOT), "Audio (*.wav *.mp3 *.flac *.ogg);;All files (*)")
        if path:
            self.open_audio(path)

    def open_audio(self, path):
        try:
            audio, sr = load_audio(str(path))
            if len(audio) == 0:
                raise ValueError("音檔沒有 samples。")
            stop_audio()
            self.audio, self.sr = audio, sr
            self.file_label.setText(f"{Path(path).name} · {len(audio)/sr:.1f}s · {sr} Hz")
            self.set_busy(False)
        except Exception as error:
            self.status.setText(f"載入失敗：{error}")

    def choose_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型設定", str(ROOT / "models"), "JSON (*.json)")
        if path:
            self.config_path = Path(path)
            self.config_label.setText(path)

    def set_busy(self, busy):
        for widget in (self.load_button, self.backend, self.config_button,
                       self.pitch_slider, self.speed_slider):
            widget.setEnabled(not busy)
        self.play_button.setEnabled(not busy and self.audio is not None)
        self.original_button.setEnabled(not busy and self.audio is not None)

    def play_original(self):
        if self.audio is not None:
            self.play_result(self.audio, self.sr)

    def play_result(self, audio, sr):
        try:
            play_audio(audio, sr)
            self.status.setText(f"播放中 · {sr} Hz · {len(audio)/sr:.1f}s")
        except Exception as error:
            self.status.setText(f"播放失敗，請檢查預設輸出裝置：{error}")

    def play_converted(self):
        if self.audio is None or self.worker is not None:
            return
        stop_audio()
        self.set_busy(True)
        arguments = dict(audio=self.audio, sr=self.sr,
                         pitch=self.pitch_slider.value(),
                         speed=self.speed_slider.value() / 100,
                         backend=self.backend.currentData(), config_path=self.config_path)
        self.worker = ConversionWorker(self.converter, arguments, self)
        self.worker.progress.connect(self.show_progress)
        self.worker.converted.connect(self.on_converted)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def show_progress(self, text):
        if self.worker is not None and not self.worker.cancel.is_set():
            self.status.setText(text)

    def on_converted(self, result):
        # Stop may arrive after the result was queued but before this slot runs.
        if self.worker is not None and not self.worker.cancel.is_set():
            self.play_result(result.audio, result.sample_rate)

    def on_failed(self, message):
        if self.worker is not None and not self.worker.cancel.is_set():
            self.status.setText(message)

    def on_finished(self):
        cancelled = self.worker.cancel.is_set()
        self.worker.deleteLater()
        self.worker = None
        self.set_busy(False)
        if cancelled:
            self.status.setText("已停止。")
        if self.closing:
            self.close()

    def stop(self):
        stop_audio()
        if self.worker is not None:
            self.worker.cancel.set()
            self.status.setText("停止播放；等待目前推論步驟結束後取消…")
        else:
            self.status.setText("已停止。")

    def closeEvent(self, event):
        self.stop()
        if self.worker is not None:
            self.closing = True
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceChanger()
    window.show()
    sys.exit(app.exec())
