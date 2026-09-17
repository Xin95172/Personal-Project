"""Qt event-loop regression tests; no audio device is needed for this suite."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import unittest
import tempfile
from pathlib import Path
from threading import Event
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from app import VoiceChanger
from voice_converter import ConvertedAudio
from test_conversion import fixture_models

QT_APP = QApplication.instance() or QApplication([])


class GuiTests(unittest.TestCase):
    def setUp(self):
        with patch('app.load_audio', return_value=(np.zeros(4000, np.float32), 16000)):
            self.window = VoiceChanger()
        self.window.show()

    def wait_finished(self):
        deadline = time.monotonic() + 20
        while self.window.worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window.worker)

    def tearDown(self):
        self.window.stop()
        self.wait_finished()
        self.window.close()

    def test_dsp_button_and_original(self):
        self.window.backend.setCurrentIndex(1)
        with patch('app.play_audio') as play:
            self.window.original_button.click()
            self.assertEqual(play.call_args.args[1], 16000)
            self.window.play_button.click()
            self.assertFalse(self.window.play_button.isEnabled())
            self.wait_finished()
            self.assertEqual(play.call_count, 2)
            self.assertTrue(self.window.play_button.isEnabled())

    def test_missing_model_is_visible_and_recovers(self):
        self.window.config_path = 'definitely-missing.json'
        self.window.play_button.click()
        self.wait_finished()
        self.assertIn('FileNotFoundError', self.window.status.text())
        self.assertTrue(self.window.play_button.isEnabled())

    def test_ai_button_reaches_onnx_and_plays_model_sample_rate(self):
        with tempfile.TemporaryDirectory() as folder, patch('app.play_audio') as play:
            self.window.config_path = fixture_models(Path(folder))
            self.window.play_button.click()
            self.wait_finished()
            play.assert_called_once()
            self.assertEqual(play.call_args.args[1], 40000)
            self.assertEqual(len(play.call_args.args[0]), 10000)

    def test_stop_discards_result_and_close_waits(self):
        entered, release = Event(), Event()
        def delayed(**kwargs):
            entered.set()
            release.wait(5)
            return ConvertedAudio(np.zeros(100), 40000)
        with patch.object(self.window.converter, 'convert', delayed), patch('app.play_audio') as play:
            self.window.play_button.click()
            self.assertTrue(entered.wait(3))
            self.window.stop_button.click()
            self.window.close()
            self.assertTrue(self.window.closing)
            release.set()
            self.wait_finished()
            play.assert_not_called()
            self.assertFalse(self.window.isVisible())


if __name__ == '__main__':
    unittest.main()
