"""Audio-owner thread and separate decoder; communicate via command queues."""
import queue
import time
import numpy as np
from PySide6.QtCore import QThread, Signal
from .audio_source import FileAudioSource
from .audio_output import AudioOutput
from .transport import TransportMixer
from .recorder import Recorder
from .devices import refresh_and_resolve


class FileLoader(QThread):
    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = queue.Queue()

    def run(self):
        while not self.isInterruptionRequested():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            key, path = job
            try:
                source = FileAudioSource(path)
                if source.pcm.shape[1] not in (1, 2):
                    raise ValueError("目前支援單聲道或雙聲道音檔")
                self.loaded.emit(key, source.pcm)
                del source
            except Exception as exc:
                self.failed.emit(key, str(exc))


class AudioRuntime(QThread):
    snapshot = Signal(object)
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.commands = queue.Queue()
        self.mixer = TransportMixer()
        self.output = None
        self.recorder = None
        self.finishing = []
        self.record_path = None
        self.record_frames = 0
        self.running = True
        self.peak = np.zeros(2)
        self.track_peaks = {}
        self.clipped = False
        self.underflows = 0

    def send(self, command, *args):
        self.commands.put((command, args))

    def stop_recording(self):
        if self.recorder:
            self.recorder.finish()
            self.finishing.append((self.recorder, self.record_path))
            self.recorder = None
            self.record_path = None

    def stop_output(self):
        self.stop_recording()
        for track in self.mixer.tracks.values():
            track.stop()
        if self.output:
            output, self.output = self.output, None
            try:
                output.stream.abort()
            finally:
                output.stream.close()

    def handle(self, command, args):
        tracks = self.mixer.tracks
        if command == "add":
            self.mixer.add(*args)
        elif command == "remove":
            self.mixer.remove(args[0])
        elif command == "controls":
            key, values = args
            if key in tracks:
                for name in ("gain", "mute", "solo", "pan"):
                    setattr(tracks[key].channel, name, values[name])
                tracks[key].loop = values["loop"]
        elif command == "master":
            self.mixer.engine.master_gain, self.mixer.engine.master_mute = args
        elif command == "start":
            if not self.output:
                selected = args[0]
                # GUI passes stable identity; CLI retains numeric device selection.
                device = refresh_and_resolve(selected) if selected is None or isinstance(selected, dict) else selected
                output = AudioOutput(device)
                try:
                    output.__enter__()
                except Exception:
                    output.stream.close()
                    raise
                self.output = output
                self.underflows = 0
                self.message.emit(f"輸出中 · {output.info['name']}")
        elif command == "transport":
            key, action = args
            targets = tracks.values() if key is None else ([tracks[key]] if key in tracks else [])
            for track in targets:
                if action == "play" and self.output:
                    track.play()
                elif action == "pause":
                    track.playing = False
                elif action == "stop":
                    track.stop()
        elif command == "seek":
            if args[0] in tracks:
                tracks[args[0]].seek(args[1])
        elif command == "record":
            if self.recorder or self.finishing:
                raise ValueError("請等上一段錄音儲存完成")
            if not self.output:
                raise ValueError("輸出未啟動，無法錄音")
            self.recorder = Recorder(args[0])
            self.record_path = args[0]
            self.record_frames = 0
            self.message.emit("錄音中 · 48 kHz / 24-bit stereo WAV")
        elif command == "record_stop":
            self.stop_recording()
        elif command == "reset_clip":
            self.clipped = False
        elif command == "stop":
            self.stop_output()
            self.message.emit("已停止輸出與所有音軌")
        elif command == "quit":
            self.running = False

    def emit_snapshot(self):
        self.snapshot.emit({
            "active": self.output is not None,
            "recording": self.record_path,
            "record_seconds": self.record_frames / 48000,
            "saving": bool(self.finishing),
            "underflows": self.underflows,
            "clip": self.clipped,
            "master": self.peak.copy(),
            "tracks": {key: {"cursor": track.cursor, "playing": track.playing,
                              "peak": self.track_peaks.get(key, np.zeros(2)).copy()}
                       for key, track in self.mixer.tracks.items()}
        })
        self.peak[:] = 0
        self.track_peaks = {}

    def run(self):
        next_update = 0
        try:
            while self.running:
                try:
                    if not self.output:
                        command, args = self.commands.get(timeout=0.03)
                        self.handle(command, args)
                        if not self.running:
                            break
                    # Bound command work so rapid UI gestures cannot starve audio.
                    for _ in range(128):
                        try:
                            command, args = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        self.handle(command, args)
                        if not self.running:
                            break
                except queue.Empty:
                    pass
                except Exception as exc:
                    self.message.emit(f"操作失敗：{exc}")
                if not self.running:
                    break
                if self.output:
                    try:
                        block = self.mixer.render()
                        self.output.write(block)
                        self.underflows = self.output.underflows
                        self.peak = np.maximum(self.peak, self.mixer.engine.master_peak)
                        self.clipped |= bool(np.any(self.mixer.engine.master_peak > 1))
                        for key, peak in self.mixer.engine.meters.items():
                            self.track_peaks[key] = np.maximum(self.track_peaks.get(key, np.zeros(2)), peak)
                        if self.recorder:
                            try:
                                self.recorder.push(block)
                                self.record_frames += len(block)
                            except Exception as exc:
                                self.message.emit(str(exc))
                                self.stop_recording()
                    except Exception as exc:
                        self.message.emit(f"音訊輸出失敗：{exc}")
                        try:
                            self.stop_output()
                        except Exception as close_error:
                            self.message.emit(f"關閉音訊裝置失敗：{close_error}")
                for recorder, path in list(self.finishing):
                    if not recorder.thread.is_alive():
                        self.message.emit(recorder.error or f"錄音已儲存：{path}")
                        self.finishing.remove((recorder, path))
                if time.monotonic() >= next_update:
                    self.emit_snapshot()
                    next_update = time.monotonic() + 0.05
        finally:
            try:
                self.stop_output()
            except Exception as exc:
                self.message.emit(f"關閉音訊裝置失敗：{exc}")
            for recorder, path in self.finishing:
                try:
                    recorder.close()
                    self.message.emit(f"錄音已儲存：{path}")
                except Exception as exc:
                    self.message.emit(str(exc))
