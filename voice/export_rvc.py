"""Export the local RVC checkpoint with dynamic attention lengths.

Requires the local rvc-export source (attentions.py must retain tensor lengths,
not int(length)), torch and onnx. Run from this directory.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import onnx
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rvc-export"))
from rvc.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from rvc_model import open_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("f0", 1) != 1:
        raise ValueError("This exporter requires an F0-enabled RVC model.")
    config = list(checkpoint["config"])
    config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    version = checkpoint.get("version", "v1")
    channels = 256 if version == "v1" else 768
    model = SynthesizerTrnMsNSFsidM(*config, is_half=False, version=version)
    incompatible = model.load_state_dict(checkpoint["weight"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("enc_q.")]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint mismatch: {missing}, {incompatible.unexpected_keys}")
    model.eval()
    def inputs(frames):
        return (torch.randn(1, frames, channels), torch.tensor([frames]),
                torch.full((1, frames), 100, dtype=torch.int64),
                torch.full((1, frames), 220.0), torch.tensor([0]),
                torch.randn(1, 192, frames))
    names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    destination = Path(args.output).resolve()
    staged = destination.with_name(destination.stem + ".pending.onnx")
    with torch.no_grad():
        torch.onnx.export(model, inputs(200), str(staged),
                          input_names=names, output_names=["audio"],
                          dynamic_axes={"phone": {1: "frames"}, "pitch": {1: "frames"},
                                        "pitchf": {1: "frames"}, "rnd": {2: "frames"},
                                        "audio": {2: "samples"}},
                          opset_version=17, dynamo=False)
    graph = onnx.load(staged)
    onnx.helper.set_model_props(graph, {"config": json.dumps(config), "version": version,
                                       "export_fix": "dynamic_relative_attention_lengths"})
    onnx.checker.check_model(graph)
    onnx.save(graph, staged)
    session = open_session(staged)
    for frames in (128, 200, 900, 174):
        output = session.run(None, dict(zip(names, [x.numpy() for x in inputs(frames)])))[0]
        assert output.size == frames * int(config[-1]) // 100, output.shape
        assert np.isfinite(output).all()
        print(f"Validated T={frames}: {output.shape}", flush=True)
    del session
    staged.replace(destination)
    print(f"Saved verified model: {destination}", flush=True)


if __name__ == "__main__":
    main()
