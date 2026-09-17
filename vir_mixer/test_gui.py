"""Integration checks. Default uses a fake output; MIXER_REAL_AUDIO=1 uses speakers."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication
from mixer.mixer_gui import MixerWindow
from mixer.session import save_scene, load_scene


class FakeOutput:
    def __init__(self, device=None):
        self.info = {"name": "Test output"}
        self.stream = self
        self.underflows = 0

    def __enter__(self):
        return self

    def write(self, block):
        time.sleep(len(block) / 48000)

    def abort(self):
        pass

    def close(self):
        pass


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump_until(self, condition, timeout=6):
        end = time.monotonic() + timeout
        while not condition() and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(.01)
        self.assertTrue(condition(), self.window.status.text())

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        if os.environ.get("MIXER_REAL_AUDIO") != "1":
            self.output_patch = patch("mixer.runtime.AudioOutput", FakeOutput)
            self.output_patch.start()
            self.addCleanup(self.output_patch.stop)
        self.window = MixerWindow()
        self.window.show()
        self.state = {}
        self.window.runtime.snapshot.connect(self.set_state)
        self.pump_until(lambda: bool(self.window.cards) and all(c.ready for c in self.window.cards.values()))

    def set_state(self, state):
        self.state = state

    def tearDown(self):
        self.window.close()
        self.pump_until(lambda: not self.window.runtime.isRunning() and not self.window.loader.isRunning())
        self.app.processEvents()
        self.window.close()

    def test_multitrack_controls_record_and_scene(self):
        window = self.window
        first = next(iter(window.cards))
        second = window.add_file(window.cards[first].file_path)
        self.pump_until(lambda: window.cards[second].ready)
        for card in window.cards.values():
            card.fader.setValue(-180)
        window.transport(None, "play")
        self.pump_until(lambda: self.state.get("tracks", {}).get(first, {}).get("cursor", 0) > 10000)
        window.cards[first].pause.click()
        self.pump_until(lambda: not self.state["tracks"][first]["playing"])
        paused_cursor = self.state["tracks"][first]["cursor"]
        second_cursor = self.state["tracks"][second]["cursor"]
        self.pump_until(lambda: self.state["tracks"][second]["cursor"] > second_cursor + 5000)
        self.assertEqual(self.state["tracks"][first]["cursor"], paused_cursor)
        window.cards[first].play.click()
        window.cards[first].solo.click()
        self.pump_until(lambda: self.state["tracks"][first]["playing"] and
                        max(self.state["tracks"][second]["peak"]) == 0)
        window.cards[first].mute.click()
        self.pump_until(lambda: max(self.state["master"]) == 0)
        window.cards[first].mute.click()
        window.cards[first].solo.click()
        window.cards[second].seek.setValue(5000)
        window.cards[second].seek.sliderReleased.emit()
        self.pump_until(lambda: self.state["tracks"][second]["cursor"] >= window.cards[second].frames // 2)
        output = self.path / "recording.wav"
        with patch("mixer.mixer_gui.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            window.record_button.click()
        self.pump_until(lambda: window.recording and self.state["record_seconds"] >= .3)
        window.record_button.click()
        self.pump_until(lambda: not window.recording and not window.saving and output.exists())
        data, rate = sf.read(output, dtype="float32", always_2d=True)
        self.assertEqual(rate, 48000)
        self.assertEqual(data.shape[1], 2)
        self.assertGreater(len(data), 10000)
        self.assertGreater(float(np.max(np.abs(data))), .0001)
        self.assertEqual(sf.info(output).subtype, "PCM_24")
        print(f"GUI recording: {len(data)} frames, underflows: {self.state['underflows']}")
        window.cards[first].name.setText("人聲音軌")
        scene_path = self.path / "scene.json"
        save_scene(scene_path, window.scene())
        window.output_stop.click()
        self.pump_until(lambda: not window.active)
        window.apply_scene(load_scene(scene_path))
        self.pump_until(lambda: all(c.ready for c in window.cards.values()))
        self.assertEqual(next(iter(window.cards.values())).name.text(), "人聲音軌")
        self.assertFalse(window.active)

    def test_missing_file_device_and_record_on_close(self):
        window = self.window
        key = window.add_file(self.path / "missing.mp3")
        self.pump_until(lambda: "載入失敗" in window.cards[key].info.text())
        self.assertFalse(window.cards[key].play.isEnabled())
        window.remove_track(key)
        window.restore_device({"name": "missing-output", "host": "missing-host"})
        self.assertFalse(window.ensure_output())
        window.devices.setCurrentIndex(0)
        window.transport(None, "play")
        self.pump_until(lambda: window.active)
        output = self.path / "close-recording.wav"
        with patch("mixer.mixer_gui.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            window.record_button.click()
        self.pump_until(lambda: window.recording and self.state["record_seconds"] >= .2)
        window.close()
        self.pump_until(lambda: not window.runtime.isRunning())
        self.assertGreater(sf.info(output).frames, 5000)


if __name__ == "__main__":
    unittest.main()
