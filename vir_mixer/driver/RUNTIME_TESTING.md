# Runtime deployment and validation (TEST machine only)

Status: Debug and Release build, Universal API/INF checks and unsigned catalog
creation pass. No VirMixer driver is installed. None of the system-changing
commands below have been executed. Runtime correctness is still unproven.

## Reproduce the package without changing Windows

From the repository root:

```powershell
$python = 'C:\Users\UUU\anaconda3\envs\xin\python.exe'
& .\driver\package.ps1 -Configuration Debug -Python $python
& .\driver\package.ps1 -Configuration Release -Python $python
& $python driver/tests/verify_source.py
& $python driver/tests/test_prepare.py -v
& $python driver/tests/test_verify_driver.py -v
```

`package.ps1` creates a new directory each time. `out/latest-package-Debug.txt`
and `out/latest-package-Release.txt` identify only the latest validated packages.
The manifest records SHA256 of precisely SYS/INF/CAT; this is reproducibility of
procedure, not a promise of byte-identical output (WDK timestamps can differ).
WDK is pinned to the installed 10.0.28000.0 tool directories.
Native C++ ring tests: `driver/test-core.ps1` in an x64 Native Tools shell, or
supply `-Zig <zig.exe>`. Runtime dependencies are numpy, scipy and sounddevice
(already available in the xin environment).

## Approval boundary and exact actions

All `target.ps1` actions require `-Apply` and an elevated PowerShell. Omitting it
stops before creating files or making system changes. Each action is separate;
no reboot, Secure Boot modification or driver installation happens implicitly.
Before boot-setting changes, ensure the test machine has a recovery path and
any required BitLocker recovery key. Prefer a disposable Windows test machine.

After explicit approval, from the repository root in elevated PowerShell:

```powershell
$package = (Get-Content .\driver\out\latest-package-Debug.txt -Raw).Trim()
.\driver\target.ps1 -Action Sign -Package $package -Apply
.\driver\target.ps1 -Action Trust -Apply
.\driver\target.ps1 -Action EnableTestMode -Apply
```

Sign creates a one-year test certificate/private key in CurrentUser\My and a
new package, embeds a SHA256 signature in SYS, regenerates the CAT over the
signed SYS, then signs the CAT. The private key is not exported. Trust installs
only that certificate into LocalMachine Root and TrustedPublisher. EnableTestMode
runs `bcdedit /set testsigning on`; this requires a restart before deployment.
If Secure Boot blocks this, stop and report it; do not disable Secure Boot or
Memory Integrity automatically. These operations expand trust for the test
certificate and enable test-signed code. They are not production signing.

After the approved restart:

```powershell
.\driver\target.ps1 -Action Install -Apply
```

Install verifies hashes and signatures/catalog membership before calling the
WDK x64 `devcon install <signed INF> Root\VirMixerAudio`. It creates a root
MEDIA devnode, imports the driver package into Driver Store and installs the
VirMixerAudio service. Duplicate devnodes are refused. It records the published
INF/device identity in `driver/out/target-state/installed.json`. A restart may
be requested (reported, never automatic). Kernel bugs may cause hangs, BSOD,
audio loss or an unbootable test system; preserve unsaved work before testing.

## Prove the PCM path

Close applications using VirMixer, set endpoint levels to 100%, turn off audio
enhancements, then run a new process. Leave physical speakers/default devices
unchanged. The test only opens explicitly matched VirMixer WASAPI endpoints.

```powershell
$python = 'C:\Users\UUU\anaconda3\envs\xin\python.exe'
& $python driver/tests/verify_driver.py --run --repeats 10 --report driver/out/runtime-shared.json
& $python driver/tests/verify_driver.py --run --exclusive --repeats 100 --report driver/out/runtime-exclusive.json
& $python driver/tests/verify_driver.py --run --exclusive --repeats 1 --long-seconds 3600 --report driver/out/runtime-hour.json
```

The test checks 48k PCM16 stereo; independent deterministic L/R broadband data;
correlation >= .98, amplitude ratio .90..1.10, channel alignment within one frame;
no per-second discontinuity greater than two frames; no PortAudio glitches;
no-source silence before/after testing; independent open/close cycles; and
continuous output for the requested long duration. Captured data is temporarily
spooled to disk (~700 MB/hour), analyzed in bounded windows, then removed.
Reported initial offset includes host buffering and scheduling; it is NOT a
calibrated end-to-end latency measurement. JSON records failures as well as passes.
A missing device fails; no silent fallback to VB-CABLE or physical devices.

Native ring tests cover actual production capacity/prebuffer, overflow dropping
oldest whole frames, underrun zero fill, re-priming, wraps and reset. These tests
prove the algorithm, not kernel scheduling. Runtime no-source capture checks
observable silence. Forced ring overflow under kernel scheduling still needs
instrumentation/debugger observation; PortAudio overflow indicates a host-side
capture overrun, not proof that the kernel ring overflowed. Do not conflate them.

## Static review record

- One ring per nonpaged adapter; 19,200 bytes capacity and 3,840-byte priming.
- Read/write/reset serialize through one spin lock; no allocations or file I/O
  in the bridge. Ring code takes no external locks, preventing a reverse lock
  dependency with stream position locks. DMA access stays in nonpageable code.
- Exactly one underlying render and capture stream; Windows shared-mode mixing
  remains outside the bridge. Both hardware sides advertise 48k/16-bit/stereo.
- Full stereo frames are preserved; insufficient capture data is initialized
  to zero. Overflow discards oldest data; no uninitialized audio is exposed.
- Source/destination DMA copying is limited to the newest DMA window when late.
- Timer deletion/drain now precedes miniport release/DPC free. The stream retains
  the miniport. SysVAD's miniport-to-adapter link is a weak reference managed by
  the parent adapter/subdevice lifetime; surprise removal and power transitions
  must still be checked on a real kernel target.
- State transitions clear the shared ring. Starting/stopping either side can
  interrupt the other side temporarily; recovery is re-primed, not gapless.
- These are source-level findings; lock correctness under stress, actual endpoint
  registration, DMA timing and PnP/power safety require runtime evidence.

## Sleep/wake and device lifecycle

After PCM passes, run playback/capture, manually sleep and wake the test machine
10 times. Also test idle sleep, closing one side, reopening both sides, and device
removal only after closing all clients. Re-run PCM and 10 start/stop cycles after
each wake. Record OS build, package hash, duration, error events, endpoint status,
crash dump location and test result. Require no stale PCM, hang, bugcheck or lost
endpoint. Do not declare a pass just because sound resumes once.

## Driver Verifier (separate approval)

Save the current configuration with `verifier /querysettings`. On the isolated
machine only, after explicit approval:

```powershell
.\driver\target.ps1 -Action VerifierOn -Apply
```

This selects standard checks for VirMixerAudio.sys only. Restart, repeat PCM,
start/stop, long-run and sleep/wake tests; collect dumps on bugcheck. To recover,
boot Safe Mode if needed, run `verifier /reset`, then restart. This reset clears
all Verifier configuration; restore any pre-existing configuration manually.
Never enable all-driver verification for this test.

## Discord / OBS

After PCM passes: Mixer output = VirMixer Input; Discord microphone = VirMixer
Output; Discord speaker = physical headphones. Use mic test before joining a
call. Disable voice isolation/noise suppression/AGC for music tests to avoid
suppressing intended content. OBS Audio Input Capture = VirMixer Output, project
rate = 48k. Record a local stereo file and inspect L/R and dropouts. Avoid routing
monitoring back into VirMixer Input (feedback). Measure audible latency separately.
Do not send audio into calls without the user's instruction.

## Rollback / recovery

Read `out/target-state/installed.json` or `pnputil /enum-drivers` to find the exact
published `oemNN.inf` for VirMixer. Replace the placeholder below with that value;
never guess another device's OEM INF. In an elevated shell:

```powershell
.\driver\target.ps1 -Action Uninstall -PublishedInf oemNN.inf -Apply
.\driver\target.ps1 -Action RemoveTrust -Apply
.\driver\target.ps1 -Action DisableTestMode -Apply
```

Uninstall verifies provider/original INF, removes only Root\VirMixerAudio and its
identified Driver Store package. RemoveTrust deletes the saved test certificate
from the two machine trust stores and CurrentUser\My (including signing access).
DisableTestMode turns testsigning off; restart to apply. If Verifier was enabled,
reset it first. Verify VirMixer devices/service no longer load, its OEM package
is absent, the saved thumbprint is absent from the stores, and testsigning is
off. If Windows cannot start normally, use Safe Mode/recovery with the exact
recorded package identity; do not delete random driver files. A pre-test VM
snapshot is the cleanest rollback. All runtime/system actions remain unexecuted.

Microsoft references:
- https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/devcon-install
- https://learn.microsoft.com/en-us/windows-hardware/drivers/install/the-testsigning-boot-configuration-option
- https://learn.microsoft.com/en-us/windows-hardware/drivers/develop/preparing-a-computer-for-manual-driver-deployment
- https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/nf-wdm-exdeletetimer
