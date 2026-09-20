"""Check generated integration invariants; this is NOT a WDK compile or INF validation."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
source = root / 'build-source/audio/sysvad'
def read(path):
    return (source / path).read_text(encoding='utf-8')

pairs = read('TabletAudioSample/minipairs.h')
for name, endpoint in [('g_RenderEndpoints', 'SpdifMiniports'), ('g_CaptureEndpoints', 'MicInMiniports')]:
    body = re.search(name + r'\[\]\s*=\s*\{(.*?)\}', pairs, re.S).group(1)
    assert re.findall(r'&([A-Za-z0-9_]+)', body) == [endpoint]
stream = read('EndpointsCommon/minwavertstream.cpp')
assert re.search(r'const ULONG cableActual\s*=\s*m_pMiniport->GetAdapterCommObj\(\)\s*->GetVirtualCable\(\)\s*->Read\(m_pDmaBuffer \+ bufferOffset, runWrite\);', stream)
assert 'UNREFERENCED_PARAMETER(cableActual);' in stream
assert 'ULONG Read(BYTE* destination, ULONG length)' in read('VirtualCable.h')
assert 'GetVirtualCable()->Write(m_pDmaBuffer + bufferOffset, runWrite)' in stream
assert 'm_ToneGenerator.GenerateSine(' not in stream
assert 'm_SaveData.WriteData(' not in stream
node = read('EndpointsCommon/MiniportAudioEngineNode.cpp')
guard = '#if defined(SYSVAD_BTH_BYPASS) || defined(SYSVAD_USB_SIDEBAND)'
assert node.count(guard) == 8
assert node.count('#endif // optional sideband implementation') == 8
for path in ['TabletAudioSample/micinwavtable.h', 'TabletAudioSample/spdifwavtable.h']:
    table = read(path)
    assert not re.search(r'SupportedDeviceFormats\[[1-9]', table)
    assert 'WAVE_FORMAT_EXTENSIBLE, 2, 48000, 192000, 4, 16' in table
    assert 'STATICGUIDOF(KSDATAFORMAT_SUBTYPE_IEC61937_DOLBY_DIGITAL)' not in table
    assert not re.search(r'#define\s+\w*(?:MIN|MAX)_SAMPLE_RATE\s+(?!48000)\d', table)
for relative in ['TabletAudioSample/TabletAudioSample.vcxproj', 'EndpointsCommon/EndpointsCommon.vcxproj']:
    project = read(relative)
    ET.fromstring(project)
    assert not re.search(r'\bSYSVAD_(?:BTH_BYPASS|USB_SIDEBAND|A2DP_SIDEBAND)\b', project)
assert '<Inf Include="VirMixerAudio.inx" />' in read('TabletAudioSample/TabletAudioSample.vcxproj')
driver_project = read('TabletAudioSample/TabletAudioSample.vcxproj')
assert '<TargetName>TabletAudioSample</TargetName>' not in driver_project
assert driver_project.count('<TargetName>VirMixerAudio</TargetName>') == 4
assert read('TabletAudioSample/VirMixerAudio.inx') == (root / 'package/VirMixerAudio.inx').read_text(encoding='utf-8')
for header in ['AudioRing.h', 'VirtualCable.h', 'StreamTrace.h', 'CableTimeline.h', 'FrameProbe.h']:
    assert read(header) == (root / 'core' / header).read_text(encoding='utf-8')
endpoints_project = read('EndpointsCommon/EndpointsCommon.vcxproj')
debug_group = re.search(r'<ItemDefinitionGroup Condition="\'\$\(Configuration\)\|\$\(Platform\)\'==\'Debug\|x64\'">(.*?)</ItemDefinitionGroup>', endpoints_project, re.S)
assert debug_group and 'VIRMIXER_DIAGNOSTICS=1' in debug_group.group(1)
release_group = re.search(r'<ItemDefinitionGroup Condition="\'\$\(Configuration\)\|\$\(Platform\)\'==\'Release\|x64\'">(.*?)</ItemDefinitionGroup>', endpoints_project, re.S)
assert release_group and 'VIRMIXER_DIAGNOSTICS=1' not in release_group.group(1)
virtual_cable = (root / 'core' / 'VirtualCable.h').read_text(encoding='utf-8')
assert 'DbgPrintEx' not in virtual_cable[:virtual_cable.index('void FlushStopped()')]
flush = virtual_cable[virtual_cable.index('void FlushStopped()'):]
assert flush.index('KeReleaseSpinLock(&lock_,irql);\n            if(!have)') < flush.index('DbgPrintEx')
assert 'KeGetCurrentIrql()!=PASSIVE_LEVEL' in flush
assert 'trace_.streams[0].state || trace_.streams[1].state' in flush
for project in (driver_project, endpoints_project):
    tree = ET.fromstring(project)
    enabled_configs = []
    for group in tree.findall('{*}ItemDefinitionGroup'):
        definitions = ''.join(x.text or '' for x in group.findall('{*}ClCompile/{*}PreprocessorDefinitions'))
        enabled = 'VIRMIXER_DIAGNOSTICS=1' in definitions
        if enabled:
            enabled_configs.append(group.attrib.get('Condition', ''))
    assert len(enabled_configs) == 1 and 'Debug|x64' in enabled_configs[0]
assert stream.count('->Position(m_bCapture,') == 2
assert 'UpdatePosition(ilQPC, 1)' in stream and 'UpdatePosition(ilQPC, 2)' in stream
assert '_this->UpdatePosition(qpc, 3)' in stream
assert stream.index('->RunClock(') < stream.index('                ExSetTimer')
timer = stream[stream.index('void\nTimerNotifyRT'):]
assert timer.index('KeSetEvent(nleCurrent->NotificationEvent, 0, 0);') < timer.index('++diagnosticSignals;')
assert timer.index('->Notification(') < timer.index('KeReleaseSpinLock(&_this->m_PositionSpinLock, oldIrql);')
assert 'TimeElapsedInMS >= _this->m_ulNotificationIntervalMs' in timer
assert 'if (!bufferCompleted && !_this->m_bEoSReceived)' in timer
assert 'm_ulDmaBufferSize/_this->m_ulNotificationsPerBuffer' in timer
assert 'AudioRing<19200, 3840>' in virtual_cable
assert 'SetUnderrunRecoveryMode(UnderrunRecoveryMode::Continue)' in virtual_cable
assert 'BeginTransfer(m_bCapture)' in stream
assert 'ExpireRender(' in stream and 'ReadAt(' in stream and 'WriteAt(' in stream
assert 'TimelineTransition(m_bCapture)' in stream and 'TimelineRun(m_bCapture' in stream
for project in (driver_project,endpoints_project):
    modes=set(re.findall(r'VIRMIXER_SHARED_TIMELINE=([01])',project))
    assert len(modes)==1
print('PASS: generated endpoint, format, bridge, project XML and package invariants')

# Timer teardown must finish before objects reachable from the callback are freed.
destructor = stream[stream.index('CMiniportWaveRTStream::~'):stream.index('} // ~CMiniportWaveRTStream')]
assert destructor.index('ExDeleteTimer') < destructor.index('KeFlushQueuedDpcs()') < destructor.index('m_pMiniport->Release()')
assert destructor.index('KeFlushQueuedDpcs()') < destructor.index('ExFreePoolWithTag( m_pDpc')
print('PASS: timer/DPC teardown precedes referenced object destruction')
