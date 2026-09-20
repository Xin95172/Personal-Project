# Project-local PortAudio capture-tail candidate

This is an isolated dependency experiment, not a Windows driver install.
It pins official PortAudio v19.7.0 and builds baseline/fixed variants.
The fixed variant preserves a whole captured packet's unconsumed suffix;
it does not change WASAPI latency, polling, or the VirMixer ring.

See [the investigation](../PORTAUDIO_TAIL_INVESTIGATION.md) for evidence,
limitations, hashes and sequential runtime commands. Do not replace the system
Python DLL or claim the first upstream frame loss is fixed by this patch.

From the repository root:

```powershell
$py = 'C:\Users\UUU\anaconda3\envs\xin\python.exe'
& .\driver\portaudio\build.ps1 -Variant baseline -Python $py
& .\driver\portaudio\build.ps1 -Variant fixed -Python $py
& $py -m unittest discover -s driver/tests -p 'test_*.py'
```

Visual Studio 2026 C++/CMake and a Windows SDK are required by the current
builder. Source generation rejects a changed upstream revision or edits to
previously generated source. No download/build outputs need to be committed;
upstream licensing remains in the copied source tree.

Native regression using Zig on PATH (or substitute its absolute executable):

```powershell
zig cc -Wall -Wextra -Werror -Idriver/out/portaudio-upstream/src/common driver/tests/portaudio_tail_test.c driver/out/portaudio-upstream/src/common/pa_ringbuffer.c -o driver/out/portaudio_tail_test.exe
& .\driver\out\portaudio_tail_test.exe
```

The test uses the real upstream ring implementation. Old capacity reproduces
both historical spacing/loss and the induced +32 variant; full-packet capacity
retains every possible suffix for a 1024-frame packet. Runtime forced stalls
can still drop packets upstream and should continue to fail strict no-loss
validation; that failure must not be hidden or reclassified as a pass.
