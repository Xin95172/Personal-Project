"""Contract tests use tiny synthetic ONNX graphs, NOT trained voice weights."""
import json
from pathlib import Path
import tempfile
import unittest
from threading import Event
from unittest.mock import patch
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from voice_converter import VoiceConverter, ConversionCancelled
from features import extract_f0


def save_graph(path, inputs, outputs, nodes, constants):
    graph = h.make_graph(nodes, path.stem, inputs, outputs,
                         [nh.from_array(np.asarray(value), name) for name, value in constants.items()])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 13)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def fixture_models(root):
    save_graph(root / 'content.onnx',
        [h.make_tensor_value_info('audio', T.FLOAT, [1, 1, 'N'])],
        [h.make_tensor_value_info('content', T.FLOAT, [1, 'F', 768])],
        [h.make_node('AveragePool', ['audio'], ['pooled'], kernel_shape=[320], strides=[320]),
         h.make_node('Transpose', ['pooled'], ['transposed'], perm=[0, 2, 1]),
         h.make_node('Tile', ['transposed', 'repeat'], ['content'])],
        {'repeat': np.array([1, 1, 768], np.int64)})
    save_graph(root / 'voice.onnx',
        [h.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
            ('phone', T.FLOAT, [1, 'F', 768]), ('phone_lengths', T.INT64, [1]),
            ('pitch', T.INT64, [1, 'F']), ('pitchf', T.FLOAT, [1, 'F']),
            ('ds', T.INT64, [1]), ('rnd', T.FLOAT, [1, 192, 'F']))],
        [h.make_tensor_value_info('out', T.FLOAT, [1, 1, 'N'])],
        [h.make_node('Unsqueeze', ['pitchf', 'axis'], ['expanded']),
         h.make_node('Tile', ['expanded', 'repeat'], ['repeated']),
         h.make_node('Reshape', ['repeated', 'shape'], ['flat']),
         h.make_node('Mul', ['flat', 'gain'], ['out'])],
        {'axis': np.array([2], np.int64), 'repeat': np.array([1, 1, 400], np.int64),
         'shape': np.array([1, 1, -1], np.int64), 'gain': np.array(0.001, np.float32)})
    config = root / 'config.json'
    config.write_text(json.dumps(dict(voice_model='voice.onnx', contentvec='content.onnx',
                                     sample_rate=40000, speaker_id=0)))
    return config


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)).astype(np.float32)

    def test_f0_pitch_and_silence(self):
        _, f0 = extract_f0(self.audio, 50, 0)
        bins, higher = extract_f0(self.audio, 50, 12)
        self.assertAlmostEqual(float(np.median(f0[f0 > 0])), 220, delta=4)
        np.testing.assert_allclose(higher, f0 * 2)
        self.assertTrue(np.all((bins >= 1) & (bins <= 255)))
        _, silence = extract_f0(np.zeros(4000), 25, 0)
        self.assertTrue(np.all(silence == 0))

    def test_validation_missing_weights_and_cancel(self):
        converter = VoiceConverter()
        with self.assertRaises(ValueError):
            converter.convert([], 16000)
        with self.assertRaises(ValueError):
            converter.convert(self.audio, 16000, speed=0)
        with self.assertRaises(FileNotFoundError):
            converter.convert(self.audio, 16000, config_path='missing-config.json')
        cancel = Event()
        cancel.set()
        with self.assertRaises(ConversionCancelled):
            converter.convert(self.audio, 16000, cancel=cancel)

    def test_real_onnx_runtime_with_synthetic_contract_graphs(self):
        with tempfile.TemporaryDirectory() as folder:
            config = fixture_models(Path(folder))
            result = VoiceConverter().convert(self.audio, 16000, config_path=config)
            self.assertEqual(result.sample_rate, 40000)
            self.assertEqual(len(result.audio), 20000)
            self.assertTrue(np.isfinite(result.audio).all())
            self.assertGreater(np.mean(result.audio), 0.15)
            # Incorrect declared output rate must not play at the wrong speed.
            data = json.loads(config.read_text())
            data['sample_rate'] = 48000
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'sample_rate'):
                VoiceConverter().convert(self.audio, 16000, config_path=config)

    def test_overlap_add_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            config = fixture_models(Path(folder))
            # Constant pitch isolates the overlap-add math from pitch estimation.
            def constant_pitch(audio, frames, semitones):
                return np.ones((1, frames), np.int64), np.full((1, frames), 220, np.float32)
            with patch('pipeline.extract_f0', constant_pitch):
                for duration in (7.95, 8.01, 16.05):
                    signal = np.tile(self.audio, 34)[:round(duration * 16000)]
                    result = VoiceConverter().convert(signal, 16000, config_path=config)
                    self.assertEqual(len(result.audio), round(duration * 40000))
                    np.testing.assert_allclose(result.audio, 0.22, atol=1e-6)

    def test_dsp_speed_and_rate(self):
        result = VoiceConverter().convert(self.audio, 16000, speed=1.25, backend='dsp')
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(len(result.audio), round(len(self.audio) / 1.25))


if __name__ == '__main__':
    unittest.main()
