"""Bounded, asynchronous WAV recording. Disk I/O never blocks playback writes."""
import queue
import threading
import soundfile as sf


class Recorder:
    def __init__(self, path):
        # Exclusive creation: never silently replace an existing recording.
        self.file = sf.SoundFile(path, mode="x", samplerate=48000, channels=2,
                                 format="WAV", subtype="PCM_24")
        self.queue = queue.Queue(maxsize=500)
        self.error = None
        self.frames = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mixer-recording", daemon=True)
        self.thread.start()

    def push(self, pcm):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.queue.put_nowait(pcm.copy())
        except queue.Full:
            self.error = "錄音磁碟寫入太慢，錄音已停止；已寫入的音訊保留。"
            raise RuntimeError(self.error)

    def _run(self):
        try:
            while not self.done.is_set() or not self.queue.empty():
                try:
                    block = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                self.file.write(block)
                self.frames += len(block)
        except Exception as exc:
            self.error = f"錄音失敗：{exc}"
        finally:
            try:
                self.file.close()
            except Exception as exc:
                self.error = f"錄音檔收尾失敗：{exc}"

    def finish(self):
        self.done.set()

    def close(self):
        self.finish()
        self.thread.join()
        if self.error:
            raise RuntimeError(self.error)
