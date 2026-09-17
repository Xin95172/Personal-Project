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
assert 'GetVirtualCable()->Read(m_pDmaBuffer + bufferOffset, runWrite)' in stream
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
for header in ['AudioRing.h', 'VirtualCable.h']:
    assert read(header) == (root / 'core' / header).read_text(encoding='utf-8')
print('PASS: generated endpoint, format, bridge, project XML and package invariants')

# Timer teardown must finish before objects reachable from the callback are freed.
destructor = stream[stream.index('CMiniportWaveRTStream::~'):stream.index('} // ~CMiniportWaveRTStream')]
assert destructor.index('ExDeleteTimer') < destructor.index('KeFlushQueuedDpcs()') < destructor.index('m_pMiniport->Release()')
assert destructor.index('KeFlushQueuedDpcs()') < destructor.index('ExFreePoolWithTag( m_pDpc')
print('PASS: timer/DPC teardown precedes referenced object destruction')
