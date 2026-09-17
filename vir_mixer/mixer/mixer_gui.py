"""Multi-file mixing console. All audio and decoding run outside the UI thread."""
from pathlib import Path
import math
import uuid
from PySide6.QtCore import Qt, QTimer, Signal, QProcess
from PySide6.QtGui import QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QSlider, QCheckBox, QScrollArea, QFrame,
)
from .runtime import AudioRuntime, FileLoader
from .session import load_scene, save_scene, validate_scene
from .devices import inventory
from .device_panel import DevicePanel, DeviceScanner


def gain_from_db(db):
    return 0.0 if db <= -60 else 10 ** (db / 20)


def clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def button(text, callback, checkable=False):
    item = QPushButton(text)
    item.setCheckable(checkable)
    item.clicked.connect(callback)
    return item


class StereoMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.levels = [-60.0, -60.0]
        self.holds = [-60.0, -60.0]
        self.setMinimumSize(86, 170)
        self.setToolTip("L / R 峰值電平（dBFS）；音軌為推桿後，MASTER 為截幅前")

    def set_levels(self, values):
        for i in range(2):
            level = max(-60, min(6, 20 * math.log10(max(float(values[i]), 1e-6))))
            self.levels[i] = max(level, self.levels[i] - 3)
            self.holds[i] = max(level, self.holds[i] - 0.4)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(QFont("Segoe UI", 8))
        top, height = 12, self.height() - 36
        for index in range(2):
            x = 8 + index * 22
            painter.fillRect(x, top, 16, height, QColor("#101821"))
            for segment in range(30):
                db = -60 + segment * 2
                color = "#ef6673" if db >= -2 else "#eabd68" if db >= -12 else "#5bcfa5"
                if self.levels[index] > db:
                    y = top + height - int((segment + 1) * height / 30)
                    painter.fillRect(x, y, 16, max(1, int(height / 30) - 2), QColor(color))
            y = top + int(-min(0, self.holds[index]) / 60 * height)
            painter.fillRect(x, y, 16, 2, QColor("#eef6ff"))
            painter.setPen(QColor("#99aabd"))
            painter.drawText(x + 3, self.height() - 5, "L" if index == 0 else "R")
        painter.setPen(QColor("#99aabd"))
        for db in (0, -12, -24, -36, -48, -60):
            painter.drawText(55, top + int(-db / 60 * height) + 4, str(db))


class TrackStrip(QFrame):
    controls_changed = Signal(str)
    action = Signal(str, str)
    seeked = Signal(str, float)
    removed = Signal(str)

    def __init__(self, key, path, number, settings=None):
        super().__init__()
        settings = settings or {}
        self.key, self.file_path = key, str(Path(path).resolve())
        self.ready, self.frames = False, 0
        self.setObjectName("strip")
        self.setFixedWidth(238)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        header = QHBoxLayout()
        index = QLabel(f"CH {number:02d}")
        index.setObjectName("eyebrow")
        header.addWidget(index)
        header.addStretch()
        header.addWidget(button("移除", lambda: self.removed.emit(key)))
        layout.addLayout(header)
        self.name = QLineEdit(settings.get("name") or Path(path).stem)
        self.name.setToolTip("音軌名稱")
        layout.addWidget(self.name)
        source = QLabel(Path(path).name)
        source.setObjectName("muted")
        source.setToolTip(self.file_path)
        source.setMaximumWidth(206)
        layout.addWidget(source)
        self.info = QLabel("正在載入…")
        self.info.setWordWrap(True)
        self.info.setMinimumHeight(32)
        layout.addWidget(self.info)
        row = QHBoxLayout()
        self.play = button("播放", lambda: self.action.emit(key, "play"))
        self.pause = button("暫停", lambda: self.action.emit(key, "pause"))
        self.stop = button("停止", lambda: self.action.emit(key, "stop"))
        for item in (self.play, self.pause, self.stop):
            row.addWidget(item)
            item.setEnabled(False)
        layout.addLayout(row)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 10000)
        self.seek.setEnabled(False)
        self.seek.setToolTip("拖曳後放開，跳至指定位置")
        self.seek.sliderReleased.connect(lambda: self.seeked.emit(key, self.seek.value() / 10000))
        layout.addWidget(self.seek)
        self.time = QLabel("00:00 / 00:00")
        self.time.setObjectName("muted")
        layout.addWidget(self.time)
        self.loop = QCheckBox("循環播放")
        self.loop.setChecked(settings.get("loop", False))
        layout.addWidget(self.loop)
        row = QHBoxLayout()
        self.mute = button("MUTE", lambda: self.controls_changed.emit(key), True)
        self.solo = button("SOLO", lambda: self.controls_changed.emit(key), True)
        self.mute.setObjectName("mute")
        self.solo.setObjectName("solo")
        self.mute.setChecked(settings.get("mute", False))
        self.solo.setChecked(settings.get("solo", False))
        self.solo.setToolTip("只輸出所有 Solo 音軌；Mute 優先。Solo 會影響主輸出與錄音。")
        row.addWidget(self.mute)
        row.addWidget(self.solo)
        layout.addLayout(row)
        self.pan_label = QLabel()
        layout.addWidget(self.pan_label)
        self.pan = QSlider(Qt.Orientation.Horizontal)
        self.pan.setRange(-100, 100)
        self.pan.setValue(int(settings.get("pan", 0)))
        self.pan.setToolTip("立體聲平衡：向左／右衰減另一邊；單聲道音檔會先複製到 L/R")
        layout.addWidget(self.pan)
        faders = QHBoxLayout()
        column = QVBoxLayout()
        self.db = QLabel()
        self.db.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.db)
        self.fader = QSlider(Qt.Orientation.Vertical)
        self.fader.setRange(-600, 120)
        self.fader.setValue(round(settings.get("gain_db", 0) * 10))
        self.fader.setMinimumHeight(170)
        self.fader.setToolTip("音量 −∞ 至 +12 dB；最低位置為完全靜音")
        column.addWidget(self.fader, 1, Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(button("0 dB", lambda: self.fader.setValue(0)))
        faders.addLayout(column, 1)
        self.meter = StereoMeter()
        faders.addWidget(self.meter, 1)
        layout.addLayout(faders, 1)
        self.peak = QLabel("峰值 −∞ dBFS")
        self.peak.setObjectName("muted")
        layout.addWidget(self.peak)
        for signal in (self.fader.valueChanged, self.pan.valueChanged, self.loop.toggled):
            signal.connect(self.changed)
        self.update_labels()

    def update_labels(self):
        db = self.fader.value() / 10
        self.db.setText("−∞ dB" if db <= -60 else f"{db:+.1f} dB")
        pan = self.pan.value()
        self.pan_label.setText("BAL · 中央" if not pan else f"BAL · {'L' if pan < 0 else 'R'} {abs(pan)}")

    def changed(self, *_):
        self.update_labels()
        self.controls_changed.emit(self.key)

    def controls(self):
        return {"gain": gain_from_db(self.fader.value() / 10), "pan": self.pan.value() / 100,
                "mute": self.mute.isChecked(), "solo": self.solo.isChecked(), "loop": self.loop.isChecked()}

    def settings(self):
        return {"path": self.file_path, "name": self.name.text(), "gain_db": self.fader.value() / 10,
                "pan": self.pan.value(), "mute": self.mute.isChecked(),
                "solo": self.solo.isChecked(), "loop": self.loop.isChecked()}

    def loaded(self, pcm):
        self.ready, self.frames = True, len(pcm)
        self.info.setText(f"48 kHz · {'單聲道' if pcm.shape[1] == 1 else '雙聲道'}")
        for item in (self.play, self.pause, self.stop, self.seek):
            item.setEnabled(True)
        self.time.setText(f"00:00 / {clock_text(self.frames / 48000)}")

    def update_state(self, state):
        if not self.seek.isSliderDown():
            self.seek.setValue(round(state["cursor"] / max(1, self.frames) * 10000))
        self.time.setText(f"{clock_text(state['cursor'] / 48000)} / {clock_text(self.frames / 48000)}")
        self.play.setText("播放中" if state["playing"] else "播放")
        self.play.setEnabled(not state["playing"])
        self.pause.setEnabled(state["playing"])
        self.meter.set_levels(state["peak"])
        peak = float(max(state["peak"]))
        self.peak.setText("峰值 −∞ dBFS" if peak < 1e-6 else f"峰值 {20 * math.log10(peak):+.1f} dBFS")


STYLE = """
QWidget { background: #111923; color: #e4edf5; font-family: 'Segoe UI', 'Microsoft JhengHei'; font-size: 12px; }
QFrame#strip { background: #1b2735; border: 1px solid #314153; border-radius: 10px; }
QFrame#strip QLabel, QFrame#strip QCheckBox { background: transparent; }
QLabel#title { font-size: 25px; font-weight: 700; }
QLabel#eyebrow { color: #67d9b2; font-weight: 700; }
QLabel#muted { color: #9caec0; }
QPushButton { background: #2a3a4c; border: 1px solid #40536a; border-radius: 5px; padding: 7px 9px; }
QPushButton:hover { background: #354d64; }
QPushButton:disabled { color: #66798b; background: #202e3c; border-color: #2a3a4c; }
QPushButton#primary { background: #287b67; border-color: #419b86; }
QPushButton#mute:checked { background: #a44252; color: white; }
QPushButton#solo:checked { background: #a78234; color: white; }
QLineEdit, QComboBox { background: #111c28; border: 1px solid #40536a; border-radius: 4px; padding: 6px; }
QSlider::groove:vertical { background: #101821; width: 7px; border-radius: 3px; }
QSlider::handle:vertical { background: #a3bacc; border: 1px solid #d0e3f1; height: 17px; margin: 0 -9px; border-radius: 3px; }
QSlider::groove:horizontal { background: #101821; height: 5px; }
QSlider::handle:horizontal { background: #72cdb5; width: 12px; margin: -4px 0; border-radius: 5px; }
QScrollArea { border: none; }
"""


class MixerWindow(QWidget):
    def __init__(self, path=None):
        super().__init__()
        self.cards = {}
        self.next_channel = 1
        self.device_inventory = []
        self.desired_device = None
        self.closing = self.active = self.recording = self.saving = False
        self.setWindowTitle("Audio Mixer · File Console")
        self.resize(1150, 880)
        self.setMinimumSize(800, 680)
        self.setStyleSheet(STYLE)
        self.runtime, self.loader = AudioRuntime(self), FileLoader(self)
        self.runtime.snapshot.connect(self.update_state)
        self.runtime.message.connect(self.show_message)
        self.loader.loaded.connect(self.file_loaded)
        self.loader.failed.connect(self.file_failed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("Audio Mixer")
        title.setObjectName("title")
        header.addWidget(title)
        subtitle = QLabel("FILE CONSOLE  /  48 kHz")
        subtitle.setObjectName("muted")
        header.addWidget(subtitle)
        header.addStretch()
        self.add_button = button("＋ 加入音檔", self.choose_files)
        self.add_button.setObjectName("primary")
        header.addWidget(self.add_button)
        header.addWidget(button("儲存設定", self.save_settings))
        self.load_button = button("載入設定", self.load_settings)
        header.addWidget(self.load_button)
        layout.addLayout(header)
        routing = QHBoxLayout()
        routing.addWidget(QLabel("主輸出"))
        self.devices = QComboBox()
        self.devices.setMinimumWidth(240)
        self.devices.currentIndexChanged.connect(self.device_choice_changed)
        routing.addWidget(self.devices, 1)
        self.refresh = button("更新裝置", self.load_devices)
        routing.addWidget(self.refresh)
        self.device_button = button('輸入／輸出裝置', self.show_devices)
        routing.addWidget(self.device_button)
        routing.addWidget(button("全部播放", lambda: self.transport(None, "play")))
        routing.addWidget(button("全部暫停", lambda: self.transport(None, "pause")))
        routing.addWidget(button("全部停止", lambda: self.transport(None, "stop")))
        self.output_stop = button("關閉輸出", lambda: self.runtime.send("stop"))
        routing.addWidget(self.output_stop)
        layout.addLayout(routing)
        hint = QLabel("Discord：主輸出選 CABLE Input，Discord 麥克風選 CABLE Output。Solo 會影響主輸出與錄音。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        body = QHBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        bank = QWidget()
        self.bank = QHBoxLayout(bank)
        self.bank.setContentsMargins(0, 0, 6, 6)
        self.bank.setSpacing(12)
        self.bank.addStretch()
        scroll.setWidget(bank)
        body.addWidget(scroll, 1)
        body.addWidget(self.make_master())
        layout.addLayout(body, 1)
        footer = QHBoxLayout()
        self.status = QLabel("加入音檔後按播放；支援拖放多個音檔。")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(self.status, 1)
        self.health = QLabel("輸出未啟動")
        self.health.setObjectName("muted")
        footer.addWidget(self.health)
        layout.addLayout(footer)
        self.setAcceptDrops(True)
        self.device_panel = DevicePanel(self)
        self.device_panel.output_selected.connect(self.select_output)
        self.scanner = DeviceScanner(self)
        self.scanner.updated.connect(self.receive_devices)
        self.scanner.failed.connect(self.show_message)
        self.device_panel.scan_requested.connect(self.load_devices)
        try:
            self.receive_devices(inventory())
        except Exception as exc:
            self.show_message(f'無法列出裝置：{exc}')
        self.scanner.timer.start()
        self.load_devices()
        self.runtime.start()
        self.loader.start()
        self.update_master()
        default = Path(path) if path else Path(__file__).resolve().parents[1] / "p0002-7.mp3"
        if default.exists():
            self.add_file(default)

    def make_master(self):
        panel = QFrame()
        panel.setObjectName("strip")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        label = QLabel("MASTER")
        label.setObjectName("eyebrow")
        layout.addWidget(label)
        layout.addWidget(QLabel("主混音 / 錄音"))
        self.master_mute = button("MASTER MUTE", self.update_master, True)
        self.master_mute.setObjectName("mute")
        layout.addWidget(self.master_mute)
        self.master_label = QLabel("−6.0 dB")
        layout.addWidget(self.master_label)
        row = QHBoxLayout()
        self.master = QSlider(Qt.Orientation.Vertical)
        self.master.setRange(-600, 60)
        self.master.setValue(-60)
        self.master.setToolTip("總音量 −∞ 至 +6 dB；最低位置為完全靜音")
        self.master.valueChanged.connect(self.update_master)
        row.addWidget(self.master, 1)
        self.master_meter = StereoMeter()
        row.addWidget(self.master_meter, 2)
        layout.addLayout(row, 1)
        self.clip = button("CLIP · 正常", lambda: self.runtime.send("reset_clip"))
        self.clip.setToolTip("超過 0 dBFS 時亮起並保持；點擊重設。請降低音軌或總音量。")
        layout.addWidget(self.clip)
        layout.addWidget(QLabel("48 kHz · 24-bit WAV"))
        self.record_label = QLabel("尚未錄音")
        self.record_label.setWordWrap(True)
        layout.addWidget(self.record_label)
        self.record_button = button("● 開始錄音", self.record)
        layout.addWidget(self.record_button)
        description = QLabel("錄下主推桿後的混音。\n靜音與 Solo 會一併錄入；\n暫停音軌時仍持續錄音。")
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        return panel

    def show_message(self, text):
        self.status.setText(text)

    def load_devices(self):
        self.scanner.scan()

    def show_devices(self):
        self.device_panel.show()
        self.device_panel.raise_()
        self.load_devices()

    def select_output(self, data):
        if not self.active:
            self.restore_device(data)
            self.show_message(f"主輸出已選擇：{data['name']}")

    def device_choice_changed(self, *_):
        if self.devices.currentIndex() >= 0:
            self.desired_device = self.devices.currentData()

    def receive_devices(self, data):
        if self.closing or (self.device_inventory == data and self.devices.count()):
            return
        self.device_inventory = data
        self.device_panel.set_inventory(data)
        self.devices.blockSignals(True)
        self.devices.clear()
        self.devices.addItem("系統預設輸出", None)
        for info in data:
            if info['outputs']:
                self.devices.addItem(f"{info['name']} · {info['host']} · {info['outputs']} ch", info)
        self.restore_device(self.desired_device)
        self.devices.blockSignals(False)
        self.device_button.setText(f"輸入 {sum(d['inputs'] > 0 for d in data)} / 輸出 {sum(d['outputs'] > 0 for d in data)}")

    def restore_device(self, desired):
        self.desired_device = desired
        if desired is None:
            self.devices.setCurrentIndex(0)
            return
        for i in range(1, self.devices.count()):
            item = self.devices.itemData(i)
            if item["name"] == desired["name"] and item["host"] == desired["host"]:
                self.devices.setCurrentIndex(i)
                return
        self.devices.setCurrentIndex(-1)
        self.show_message("原輸出裝置已不在清單中，請重新選擇主輸出。")

    def ensure_output(self):
        if self.devices.currentIndex() < 0:
            self.show_message("請先選擇主輸出裝置。")
            return False
        data = self.devices.currentData()
        self.runtime.send("start", data)
        return True

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "加入音檔", "", "音訊 (*.mp3 *.wav *.flac *.ogg);;所有檔案 (*)")
        for path in paths:
            self.add_file(path)

    def add_file(self, path, settings=None):
        if len(self.cards) >= 32:
            self.show_message("目前最多 32 條音軌；請移除不需要的音軌。")
            return
        key = uuid.uuid4().hex
        card = TrackStrip(key, path, self.next_channel, settings)
        self.next_channel += 1
        self.cards[key] = card
        self.bank.insertWidget(self.bank.count() - 1, card)
        card.controls_changed.connect(self.update_controls)
        card.action.connect(self.transport)
        card.seeked.connect(lambda k, f: self.runtime.send("seek", k, f))
        card.removed.connect(self.remove_track)
        self.loader.jobs.put((key, str(path)))
        return key

    def file_loaded(self, key, pcm):
        if key not in self.cards or self.closing:
            return
        self.runtime.send("add", key, pcm)
        self.cards[key].loaded(pcm)
        self.update_controls(key)

    def file_failed(self, key, error):
        if key in self.cards:
            self.cards[key].info.setText("載入失敗 · 移除此音軌後重試")
            self.cards[key].info.setToolTip(error)
            self.show_message(f"音檔載入失敗：{error}")

    def remove_track(self, key):
        card = self.cards.pop(key, None)
        if card:
            self.runtime.send("remove", key)
            self.bank.removeWidget(card)
            card.deleteLater()

    def update_controls(self, key):
        if key in self.cards and self.cards[key].ready:
            self.runtime.send("controls", key, self.cards[key].controls())

    def update_master(self, *_):
        db = self.master.value() / 10
        self.master_label.setText("−∞ dB" if db <= -60 else f"{db:+.1f} dB")
        self.runtime.send("master", gain_from_db(db), self.master_mute.isChecked())

    def transport(self, key, action):
        if action == "play" and not self.ensure_output():
            return
        self.runtime.send("transport", key, action)

    def record(self):
        if self.recording:
            self.runtime.send("record_stop")
            return
        path, _ = QFileDialog.getSaveFileName(self, "錄製主混音（請使用新檔名）", "mix.wav", "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        if Path(path).exists():
            self.show_message("錄音不覆蓋現有檔案，請選擇新的檔名。")
            return
        if self.ensure_output():
            self.runtime.send("record", path)

    def update_state(self, state):
        self.active, self.recording, self.saving = state["active"], bool(state["recording"]), state["saving"]
        self.devices.setEnabled(not self.active)
        self.refresh.setEnabled(True)
        self.device_panel.output_active = self.active
        self.device_panel.selection_changed()
        self.load_button.setEnabled(not self.active and not self.saving)
        self.output_stop.setEnabled(self.active)
        self.record_button.setEnabled(not self.saving)
        self.record_button.setText("■ 停止錄音" if self.recording else "● 開始錄音")
        if self.recording:
            self.record_label.setText(f"REC  {clock_text(state['record_seconds'])}")
            self.record_label.setToolTip(state["recording"])
        else:
            self.record_label.setText("正在儲存…" if self.saving else "錄音待命")
        self.health.setText(("輸出中" if self.active else "輸出未啟動") + f" · 欠載 {state['underflows']}")
        self.master_meter.set_levels(state["master"])
        self.clip.setText("CLIP · 請降低音量" if state["clip"] else "CLIP · 正常")
        self.clip.setStyleSheet("background: #a44252;" if state["clip"] else "")
        for key, data in state["tracks"].items():
            if key in self.cards:
                self.cards[key].update_state(data)

    def scene(self):
        device = self.devices.currentData()
        return {"version": 1, "master_db": self.master.value() / 10,
                "master_mute": self.master_mute.isChecked(),
                "device": {"name": device["name"], "host": device["host"]} if device else None,
                "tracks": [card.settings() for card in self.cards.values()]}

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "儲存混音設定", "mixer-scene.json", "JSON (*.json)")
        if path:
            try:
                save_scene(path, self.scene())
                self.show_message(f"設定已儲存：{path}")
            except Exception as exc:
                self.show_message(f"無法儲存設定：{exc}")

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "載入混音設定", "", "JSON (*.json)")
        if path:
            try:
                self.apply_scene(load_scene(path))
            except Exception as exc:
                self.show_message(f"無法載入設定：{exc}")

    def apply_scene(self, scene):
        validate_scene(scene)
        if self.active or self.saving:
            raise ValueError("請先關閉輸出並等候錄音儲存完成")
        for key in list(self.cards):
            self.remove_track(key)
        self.next_channel = 1
        self.master.setValue(round(scene.get("master_db", -6) * 10))
        self.master_mute.setChecked(scene.get("master_mute", False))
        self.update_master()
        self.show_message("設定已載入；音檔讀取完成後可開始播放。")
        self.restore_device(scene.get("device"))
        for settings in scene["tracks"]:
            self.add_file(settings["path"], settings)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.closing:
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.add_file(url.toLocalFile())
        event.acceptProposedAction()

    def closeEvent(self, event):
        if self.runtime.isRunning() or self.loader.isRunning() or self.scanner.process.state() != QProcess.ProcessState.NotRunning:
            event.ignore()
            if not self.closing:
                self.closing = True
                self.setEnabled(False)
                self.show_message("正在關閉音訊並完成錄音儲存…")
                self.loader.requestInterruption()
                self.scanner.stop()
                self.runtime.send("quit")
                self.close_timer = QTimer(self)
                self.close_timer.timeout.connect(self.finish_close)
                self.close_timer.start(100)
        else:
            self.scanner.stop()
            self.runtime.wait()
            self.loader.wait()
            event.accept()

    def finish_close(self):
        if (not self.runtime.isRunning() and not self.loader.isRunning()
                and self.scanner.process.state() == QProcess.ProcessState.NotRunning):
            self.close_timer.stop()
            self.close()


def launch_gui(path=None):
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window = MixerWindow(path)
    window.show()
    return app.exec()
