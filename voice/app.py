import sys

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QSlider,
    QPushButton
)

from PySide6.QtCore import Qt

from audio_engine import (
    load_audio,
    convert_voice,
    play_audio,
    stop_audio
)

AUDIO_PATH = 'p0002-7.mp3'

class VoiceChanger(QWidget):
    def __init__(self):
        super().__init__()

        self.audio, self.sr = load_audio(AUDIO_PATH)
        self.setWindowTitle("Voice Changer")
        self.resize(400, 250)

        layout = QVBoxLayout()

        # pitch 數值文字
        self.pitch_label = QLabel("pitch : 0 st")

        # pitch slider
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(-12)
        self.pitch_slider.setMaximum(12)
        self.pitch_slider.setValue(0)

        # play button
        self.original_button = QPushButton("Play Original")
        self.play_button = QPushButton("Play Converted")
        self.stop_button = QPushButton("Stop")

        # layout
        layout.addWidget(self.pitch_label)
        layout.addWidget(self.pitch_slider)
        layout.addWidget(self.original_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        # slider connect
        self.pitch_slider.valueChanged.connect(
            self.update_pitch_label
        )

        self.original_button.clicked.connect(
            self.play_original
        )

        self.play_button.clicked.connect(
            self.play_converted
        )

        self.stop_button.clicked.connect(
            stop_audio
        )

    def update_pitch_label(self, value):
        self.pitch_label.setText(
            f"pitch: {value} st"
        )

    def play_original(self):
        play_audio(
            self.audio,
            self.sr
        )

    def play_converted(self):
        pitch = self.pitch_slider.value()
        converted = convert_voice(
            self.audio,
            self.sr,
            pitch
        )

        play_audio(
            converted,
            self.sr
        )

app = QApplication(sys.argv)
window = VoiceChanger()
window.show()
sys.exit(app.exec())