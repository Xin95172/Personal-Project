"""Audio endpoint inventory. Run in a fresh process to detect hotplug changes."""
import json
import sounddevice as sd


def inventory():
    hosts = sd.query_hostapis()
    default_input, default_output = sd.default.device
    return [dict(index=i, name=d['name'], host=hosts[d['hostapi']]['name'],
                 inputs=int(d['max_input_channels']), outputs=int(d['max_output_channels']),
                 sample_rate=float(d['default_samplerate']),
                 default_input=i == default_input, default_output=i == default_output)
            for i, d in enumerate(sd.query_devices())]


def refresh_and_resolve(selection):
    """Only call from the audio owner with no open PortAudio streams.

    PortAudio caches endpoint enumeration. Its Python binding has no public
    refresh API; this paired reinitialization is confined to stopped output.
    Cross-process device indexes must never be used as persistent identities.
    """
    sd._terminate()
    sd._initialize()
    if selection is None:
        return None
    matches = [d for d in inventory() if d['name'] == selection['name']
               and d['host'] == selection['host'] and d['outputs'] > 0]
    if not matches:
        raise ValueError("輸出裝置已移除，請重新掃描並選擇裝置")
    if len(matches) != 1:
        raise ValueError("有多個同名輸出裝置，請選擇能唯一辨識裝置的其他音訊介面")
    return matches[0]['index']


if __name__ == '__main__':
    print(json.dumps(inventory(), ensure_ascii=True))
