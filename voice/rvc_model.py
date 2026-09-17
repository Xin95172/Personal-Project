"""A small adapter for the six-input RVC ONNX export (not a model reimplementation)."""
import json
import numpy as np
import onnxruntime as ort


def open_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=options,
                               providers=["CPUExecutionProvider"])


class RVCModel:
    def __init__(self, config):
        self.session = open_session(config.voice_model)
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        expected = {"phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"}
        if set(self.inputs) != expected:
            raise ValueError(f"不相容的 RVC ONNX 輸入：{list(self.inputs)}。\n需要六輸入 phone/phone_lengths/pitch/pitchf/ds/rnd 匯出格式。")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "config" in metadata:
            exported = json.loads(metadata["config"])
            if int(exported[-1]) != config.sample_rate:
                raise ValueError("config.json 的 sample_rate 與模型內建取樣率不一致。")
            if config.speaker_id >= int(exported[-3]):
                raise ValueError("speaker_id 超出模型的說話者數量。")
        self.speaker_id = config.speaker_id
        self.sample_rate = config.sample_rate

    def infer(self, content, pitch, pitch_hz):
        frames = content.shape[1]
        channels = self.inputs["phone"].shape[-1]
        if isinstance(channels, int) and channels != content.shape[-1]:
            raise ValueError(f"RVC 需要 {channels} 維特徵，但 ContentVec 提供 {content.shape[-1]} 維。")
        values = {
            "phone": content,
            "phone_lengths": np.array([frames], dtype=np.int64),
            "pitch": pitch,
            "pitchf": pitch_hz,
            "ds": np.array([self.speaker_id], dtype=np.int64),
            "rnd": np.random.default_rng().standard_normal((1, 192, frames)),
        }
        types = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                 "tensor(int64)": np.int64}
        feed = {name: np.asarray(value, dtype=types[self.inputs[name].type])
                for name, value in values.items()}
        audio = np.asarray(self.session.run(None, feed)[0], dtype=np.float32).reshape(-1)
        expected_length = frames * self.sample_rate // 100
        if abs(len(audio) - expected_length) > self.sample_rate // 100:
            raise ValueError("模型輸出長度與 sample_rate 不一致；請確認模型的原生取樣率。")
        return audio
