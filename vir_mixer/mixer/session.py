"""JSON scene persistence: settings only, with no automatic playback or recording."""
import json
import math
from pathlib import Path


def validate_scene(scene):
    if not isinstance(scene, dict) or scene.get("version") != 1:
        raise ValueError("不支援的設定檔版本")
    tracks = scene.get("tracks")
    if not isinstance(tracks, list) or len(tracks) > 32:
        raise ValueError("設定檔最多可包含 32 條音軌")
    for track in tracks:
        if not isinstance(track, dict) or not isinstance(track.get("path"), str):
            raise ValueError("音軌缺少音檔路徑")
        if not isinstance(track.get("name", ""), str):
            raise ValueError("音軌名稱必須是文字")
        for key, low, high in (("gain_db", -60, 12), ("pan", -100, 100)):
            value = track.get(key, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"音軌 {key} 超出範圍")
        for key in ("mute", "solo", "loop"):
            if not isinstance(track.get(key, False), bool):
                raise ValueError(f"音軌 {key} 必須是布林值")
    master = scene.get("master_db", -6)
    if not isinstance(master, (int, float)) or not math.isfinite(master) or not -60 <= master <= 6:
        raise ValueError("總音量超出範圍")
    if not isinstance(scene.get("master_mute", False), bool):
        raise ValueError("總靜音設定無效")
    device = scene.get("device")
    if device is not None and (not isinstance(device, dict) or
                               not all(isinstance(device.get(k), str) for k in ("name", "host"))):
        raise ValueError("輸出裝置設定無效")
    return scene


def load_scene(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("設定檔過大")
    scene = validate_scene(json.loads(path.read_text(encoding="utf-8")))
    for track in scene["tracks"]:
        source = Path(track["path"])
        if not source.is_absolute():
            track["path"] = str((path.parent / source).resolve())
    return scene


def save_scene(path, scene):
    validate_scene(scene)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
