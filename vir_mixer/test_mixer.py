import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from mixer import Channel, MixerEngine
from mixer.audio_source import FileAudioSource
from mixer.audio_output import adapt_channels
from mixer.transport import TransportMixer, Track
from mixer.recorder import Recorder
from mixer.session import save_scene, load_scene, validate_scene


class MixerTests(unittest.TestCase):
    def test_mix_controls_and_input_preservation(self):
        pcm = np.full((4, 2), 0.5, dtype=np.float32)
        engine = MixerEngine({"a": Channel(0.5), "b": Channel(1, True)}, 0.5)
        np.testing.assert_allclose(engine.process({"a": pcm, "b": pcm}), 0.125)
        engine.channels["b"].mute = False
        np.testing.assert_allclose(engine.process({"a": pcm, "b": pcm}), 0.375)
        engine.master_gain = 4
        np.testing.assert_array_equal(engine.process({"a": pcm, "b": pcm}), 1)
        np.testing.assert_array_equal(pcm, 0.5)

    def test_resampling_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.wav"
            sf.write(path, np.full(4410, 0.1, dtype=np.float32), 44100)
            source = FileAudioSource(path)
            blocks = list(source.blocks(700))
            self.assertEqual(sum(map(len, blocks)), 4800)
            self.assertEqual(len(blocks[-1]), 600)
            np.testing.assert_array_equal(np.concatenate(blocks), source.pcm)
            self.assertEqual(source.pcm.dtype, np.float32)

    def test_channel_adaptation(self):
        stereo = np.array([[0.2, 0.6]], dtype=np.float32)
        np.testing.assert_allclose(adapt_channels(stereo, 1), [[0.4]])
        np.testing.assert_allclose(adapt_channels(stereo[:, :1], 2), [[0.2, 0.2]])

    def test_solo_and_mute_priority(self):
        pcm = np.full((4, 2), .25, dtype=np.float32)
        engine = MixerEngine({"a": Channel(solo=True), "b": Channel()})
        np.testing.assert_allclose(engine.process({"a": pcm, "b": pcm}), .25)
        engine.channels["a"].mute = True
        np.testing.assert_array_equal(engine.process({"a": pcm, "b": pcm}), 0)
        engine.channels["b"].solo = True
        np.testing.assert_allclose(engine.process({"a": pcm, "b": pcm}), .25)

    def test_balance_master_mute_and_clip_meter(self):
        pcm = np.full((4, 2), .8, dtype=np.float32)
        engine = MixerEngine({"a": Channel(gain=2, pan=-1)})
        output = engine.process({"a": pcm})
        np.testing.assert_array_equal(output[:, 1], 0)
        np.testing.assert_array_equal(output[:, 0], 1)
        np.testing.assert_allclose(engine.master_peak, [1.6, 0])
        engine.master_mute = True
        np.testing.assert_array_equal(engine.process({"a": pcm}), 0)

    def test_gain_changes_ramp_over_one_block(self):
        pcm = np.ones((480, 2), dtype=np.float32)
        engine = MixerEngine({"a": Channel()}, smoothing=True)
        engine.process({"a": pcm})
        engine.channels["a"].gain = 0
        result = engine.process({"a": pcm})
        np.testing.assert_allclose(result[0], 1)
        np.testing.assert_allclose(result[-1], 0)
        self.assertLess(np.max(np.abs(np.diff(result[:, 0]))), .003)

    def test_independent_transports_and_tail(self):
        mixer = TransportMixer()
        mixer.engine.smoothing = False
        mixer.engine.master_gain = 1
        mixer.add("a", np.full((5, 1), .1, dtype=np.float32))
        mixer.add("b", np.full((9, 2), .2, dtype=np.float32))
        mixer.tracks["a"].play()
        mixer.tracks["b"].play()
        np.testing.assert_allclose(mixer.render(4), .3)
        mixer.tracks["a"].playing = False
        np.testing.assert_allclose(mixer.render(4), .2)
        self.assertEqual(mixer.tracks["a"].cursor, 4)
        self.assertEqual(mixer.tracks["b"].cursor, 8)
        mixer.tracks["a"].play()
        block = mixer.render(4)
        np.testing.assert_allclose(block[0], .3)
        np.testing.assert_array_equal(block[1:], 0)
        self.assertFalse(mixer.tracks["a"].playing)
        self.assertFalse(mixer.tracks["b"].playing)

    def test_loop_shorter_than_block_and_seek(self):
        pcm = np.repeat(np.array([[.1], [.2], [.3]], dtype=np.float32), 2, axis=1)
        track = Track(pcm, loop=True)
        track.play()
        np.testing.assert_allclose(track.pull(8)[:, 0], [.1, .2, .3, .1, .2, .3, .1, .2])
        track.seek(1)
        np.testing.assert_allclose(track.pull(1), [[.1, .1]])
        track.stop()
        self.assertEqual(track.cursor, 0)
        np.testing.assert_array_equal(track.pull(3), 0)

    def test_remove_solo_restores_other_tracks(self):
        mixer = TransportMixer()
        mixer.engine.smoothing = False
        mixer.engine.master_gain = 1
        for key in ("a", "b"):
            mixer.add(key, np.full((20, 2), .1, dtype=np.float32))
            mixer.tracks[key].play()
        mixer.tracks["a"].channel.solo = True
        mixer.remove("a")
        np.testing.assert_allclose(mixer.render(4), .1)
        mixer.remove("b")
        np.testing.assert_array_equal(mixer.render(4), 0)

    def test_recording_exact_mix_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mix.wav"
            mixer = TransportMixer()
            for key, value in (("a", .2), ("b", -.05)):
                mixer.add(key, np.full((960, 2), value, dtype=np.float32))
                mixer.tracks[key].play()
            blocks = [mixer.render(), mixer.render()]
            recorder = Recorder(path)
            for block in blocks:
                recorder.push(block)
            recorder.close()
            result, rate = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(rate, 48000)
            self.assertEqual(sf.info(path).subtype, "PCM_24")
            np.testing.assert_allclose(result, np.concatenate(blocks), atol=2e-7)
            before = path.read_bytes()
            with self.assertRaises(Exception):
                Recorder(path)
            self.assertEqual(path.read_bytes(), before)

    def test_scene_roundtrip_and_invalid_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.json"
            scene = {"version": 1, "tracks": [{"path": "a.wav", "gain_db": -12,
                                               "solo": True, "loop": True}], "master_db": -6}
            save_scene(path, scene)
            result = load_scene(path)
            self.assertEqual(result["tracks"][0]["path"], str(Path(folder) / "a.wav"))
            self.assertTrue(result["tracks"][0]["solo"])
            scene["master_db"] = float("nan")
            with self.assertRaises(ValueError):
                validate_scene(scene)


if __name__ == "__main__":
    unittest.main()
