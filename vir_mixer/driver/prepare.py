"""Generate our SysVAD derivative from a pinned Microsoft revision. No installation."""
from pathlib import Path
import re
import shutil
import subprocess
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
REVISION = '97429c5623590d52f001249460daf43e6749d777'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Upstream context mismatch: {old[:100]}')
    return text.replace(old, new, 1)


def format_stream_indentation(text):
    """Keep injected capture calls aligned without altering local diagnostics."""
    return (text.replace(
        '                    m_pMiniport->GetAdapterCommObj()\n'
        '                        ->GetVirtualCable()\n'
        '                        ->Read(m_pDmaBuffer + bufferOffset, runWrite);',
        '            m_pMiniport->GetAdapterCommObj()\n'
        '                ->GetVirtualCable()\n'
        '                ->Read(m_pDmaBuffer + bufferOffset, runWrite);')
        .replace('                UNREFERENCED_PARAMETER(cableActual);',
                 '        UNREFERENCED_PARAMETER(cableActual);'))


def array_body(text, name, body):
    match = re.search(r'\b' + re.escape(name) + r'\[\]\s*=\s*\{', text)
    if not match:
        raise ValueError(f'Array not found: {name}')
    start, depth, end = match.end() - 1, 1, match.end()
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[:start] + '{\n' + body + '\n}' + text[end:]


def guard_optional_sideband(text):
    """Keep normal volume/mute behavior when optional sideband declarations are absent.

    Pinned upstream compiles these eight branches unconditionally because its
    projects always enable sideband. Match minwavert.h's declaration guard.
    """
    pattern = re.compile(r'    if \(IsSidebandDevice\(\) && m_pSidebandDevice->Is(?:Volume|Mute)Supported\(m_DeviceType\)\)\s*\{')
    matches = list(pattern.finditer(text))
    if len(matches) != 8:
        raise ValueError(f'Expected 8 optional sideband branches, found {len(matches)}')
    for match in reversed(matches):
        depth, end = 1, match.end()
        while depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        otherwise = re.match(r'\s*else\b', text[end:])
        if otherwise is not None:
            end += otherwise.end()
        elif 'return m_pSidebandDevice->SetMute' not in text[match.start():end]:
            raise ValueError('Unexpected sideband branch without fallback')
        text = (text[:match.start()] + '#if defined(SYSVAD_BTH_BYPASS) || defined(SYSVAD_USB_SIDEBAND)\n'
                + text[match.start():end] + '\n#endif // optional sideband implementation' + text[end:])
    return text


FORMAT = '''    {
        { sizeof(KSDATAFORMAT_WAVEFORMATEXTENSIBLE), 0, 0, 0,
          STATICGUIDOF(KSDATAFORMAT_TYPE_AUDIO), STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM),
          STATICGUIDOF(KSDATAFORMAT_SPECIFIER_WAVEFORMATEX) },
        { { WAVE_FORMAT_EXTENSIBLE, 2, 48000, 192000, 4, 16,
            sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX) },
          16, KSAUDIO_SPEAKER_STEREO, STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM) }
    }'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-timeline', action='store_true', help='Generate experimental B; default is baseline A')
    parser.add_argument('--frame-probe', action='store_true', help='Debug-only encoded PCM boundary observer (requires B)')
    parser.add_argument('--refresh-generated', action='store_true', help='Replace generated source files; preserve your edits elsewhere first')
    parser.add_argument('--output-dir', type=Path, help='Optional generation directory for reproducibility checks')
    args = parser.parse_args()
    if args.frame_probe and not args.shared_timeline:
        parser.error('--frame-probe requires --shared-timeline')
    upstream = ROOT / 'upstream'
    if not upstream.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/microsoft/Windows-driver-samples.git', str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'sparse-checkout', 'set', 'audio/sysvad'], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise ValueError(f'Expected upstream {REVISION}, found {actual}')
    destination = args.output_dir.resolve() if args.output_dir else ROOT / 'build-source'
    manifest_path = destination / '.virmixer-generated.json'
    if destination.exists() and not args.refresh_generated:
        if not manifest_path.exists():
            raise FileExistsError('Existing source has no generation manifest; preserve edits before using --refresh-generated')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        modified = [name for name, digest in manifest.items()
                    if not (destination / name).is_file()
                    or hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest]
        if modified:
            raise ValueError('Generated files were changed; preserve edits before regenerating: ' + ', '.join(modified[:10]))
    base = destination / 'audio' / 'sysvad'
    shutil.copytree(upstream / 'audio' / 'sysvad', base, dirs_exist_ok=True)
    shutil.copy2(upstream / 'LICENSE', destination / 'LICENSE-Microsoft')
    for header in ('AudioRing.h', 'VirtualCable.h', 'StreamTrace.h', 'CableTimeline.h', 'FrameProbe.h'):
        shutil.copy2(ROOT / 'core' / header, base / header)

    def edit(relative, transform):
        path = base / relative
        original = path.read_text(encoding='utf-8-sig')
        path.write_text(transform(original), encoding='utf-8')

    edit('common.h', lambda s: replace_once(s, 'DECLARE_INTERFACE_(IAdapterCommon, IUnknown)',
        '#include "VirtualCable.h"\n\nDECLARE_INTERFACE_(IAdapterCommon, IUnknown)').replace(
        '    STDMETHOD_(NTSTATUS,        Init)',
        '    virtual VirtualCable* GetVirtualCable() = 0;\n\n    STDMETHOD_(NTSTATUS,        Init)', 1))
    edit('common.cpp', lambda s: replace_once(replace_once(replace_once(s,
        '        PSERVICEGROUP           m_pServiceGroupWave;',
        '        VirtualCable m_VirtualCable;\n        PSERVICEGROUP           m_pServiceGroupWave;'),
        '        // IAdapterCommon methods',
        '        VirtualCable* GetVirtualCable() override { return &m_VirtualCable; }\n\n        // IAdapterCommon methods'),
        '    DPF_ENTER(("[CAdapterCommon::Init]"));',
        '    DPF_ENTER(("[CAdapterCommon::Init]"));\n    m_VirtualCable.Initialize();'))

    def stream(s):
        # Stop callback production and drain queued work before freeing anything
        # a timer/DPC can dereference, including the miniport and cable owner.
        destructor_end = s.index('} // ~CMiniportWaveRTStream')
        timer_start = s.index('    if (m_pNotificationTimer)', 0, destructor_end)
        timer_end = s.index('    KeFlushQueuedDpcs();', timer_start) + len('    KeFlushQueuedDpcs();')
        shutdown = s[timer_start:timer_end]
        s = s[:timer_start] + s[timer_end:]
        anchor = s.index('    PAGED_CODE();', s.index('CMiniportWaveRTStream::~')) + len('    PAGED_CODE();')
        s = s[:anchor] + '\n' + shutdown + '\n' + s[anchor:]
        s = replace_once(s, '    m_ulDmaMovementRate = pWfEx->nAvgBytesPerSec;', '''    // The bridge has one canonical hardware format; shared-mode Windows audio
    // performs client sample-rate/float conversions outside the driver.
    if (pWfEx->nSamplesPerSec != 48000 || pWfEx->nChannels != 2 ||
        pWfEx->wBitsPerSample != 16 || pWfEx->nBlockAlign != 4 ||
        pWfEx->nAvgBytesPerSec != 192000)
        return STATUS_NOT_SUPPORTED;
    m_ulDmaMovementRate = pWfEx->nAvgBytesPerSec;''')
        s = replace_once(
            s,
            '            m_ToneGenerator.GenerateSine(m_pDmaBuffer + bufferOffset, runWrite);',
            '''        const ULONG cableActual =
                    m_pMiniport->GetAdapterCommObj()
                        ->GetVirtualCable()
                        ->Read(m_pDmaBuffer + bufferOffset, runWrite);
                UNREFERENCED_PARAMETER(cableActual);'''
        )
        s = replace_once(s, '        m_SaveData.WriteData(m_pDmaBuffer + bufferOffset, runWrite);',
            '        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Write(m_pDmaBuffer + bufferOffset, runWrite);')
        s = replace_once(s, '''        if (!g_DoNotCreateDataFiles)
        {
            // Read from buffer and write to a file.
            ReadBytes(ByteDisplacement);
        }''', '''        // Always route render PCM to the capture endpoint. Never save kernel audio to disk.
        ReadBytes(ByteDisplacement);''')
        s = s.replace('    ULONG bufferOffset = m_ullLinearPosition % m_ulDmaBufferSize;', '''    // If the timer fell behind, only the most recent DMA window is valid.
    const ULONG skipped = ByteDisplacement > m_ulDmaBufferSize ? ByteDisplacement - m_ulDmaBufferSize : 0;
    ULONG bufferOffset = (m_ullLinearPosition + skipped) % m_ulDmaBufferSize;
    ByteDisplacement -= skipped;''')
        s = replace_once(s, '    switch (State_)', '''    if (State_ != m_KsState)
        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Reset();
    switch (State_)''')
        def debug(code):
            return '\n#if defined(VIRMIXER_DIAGNOSTICS) && VIRMIXER_DIAGNOSTICS\n' + code + '\n#endif\n'
        s = replace_once(s, '    if (State_ != m_KsState)', debug('''    const auto diagnosticOldState = m_KsState;
    m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->State(m_bCapture, m_KsState, State_, false,
        reinterpret_cast<ULONG_PTR>(this), 0, 0);''') + '    if (State_ != m_KsState)')
        s = replace_once(s, '    m_KsState = State_;', '    m_KsState = State_;' + debug('''    m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->State(m_bCapture, diagnosticOldState, State_, true,
        reinterpret_cast<ULONG_PTR>(this), 0, 0);
    if (State_ == KSSTATE_STOP) m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->FlushStopped();'''))
        anchor = '            m_ullLastDPCTimeStamp = m_ullDmaTimeStamp = KSCONVERT_PERFORMANCE_TIME(m_ullPerformanceCounterFrequency.QuadPart, ullPerfCounterTemp);'
        s = replace_once(s, anchor, anchor + debug('''            m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->RunClock(m_bCapture,
                ullPerfCounterTemp.QuadPart, m_ullPerformanceCounterFrequency.QuadPart,
                m_ullDmaTimeStamp, m_ullLinearPosition, m_hnsElapsedTimeCarryForward);'''))
        s = replace_once(s, '\n    qpc = KeQueryPerformanceCounter(&qpcFrequency);',
            '\n    qpc = KeQueryPerformanceCounter(&qpcFrequency);' + debug('''    _this->m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Timer(_this->m_bCapture,
        qpc.QuadPart, _this->m_hnsDPCTimeCarryForward);'''))
        marker = 'VOID CMiniportWaveRTStream::UpdatePosition\n(\n    _In_ LARGE_INTEGER ilQPC\n)\n{'
        s = replace_once(s, marker, marker + '\n    UNREFERENCED_PARAMETER(diagnosticOrigin);' + debug('    const auto diagnosticCarryIn = m_hnsElapsedTimeCarryForward;'))
        position = debug('''        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Position(m_bCapture,
            ilQPC.QuadPart, m_ullDmaTimeStamp, hnsCurrentTime, diagnosticCarryIn,
            m_hnsElapsedTimeCarryForward, m_ullLinearPosition, ByteDisplacement,
            m_ulDmaBufferSize, m_ulNotificationIntervalMs, m_ulNotificationsPerBuffer,
            m_KsState, diagnosticOrigin, m_bEoSReceived);''')

        render_position = debug('''        if (m_ulNotificationsPerBuffer > 0)
        {
            const LONG dmaRead =
                static_cast<LONG>(m_ullWritePosition % m_ulDmaBufferSize);

            const LONG osWrite =
                static_cast<LONG>(m_ulCurrentWritePosition);

            LONG distance = osWrite - dmaRead;

            const LONG halfBuffer =
                static_cast<LONG>(m_ulDmaBufferSize / 2);

            if (distance > halfBuffer)
            {
                distance -= static_cast<LONG>(m_ulDmaBufferSize);
            }
            else if (distance < -halfBuffer)
            {
                distance += static_cast<LONG>(m_ulDmaBufferSize);
            }

            DPF(D_TERSE, (
                "VirMixer: RENDERPOS qpc=%lld linear=%llu dmaRead=%ld osWrite=%ld "
                "distance=%ld disp=%lu dmaSize=%lu packet=%lld lastOsWrite=%lu",
                ilQPC.QuadPart,
                m_ullLinearPosition,
                dmaRead,
                osWrite,
                distance,
                ByteDisplacement,
                m_ulDmaBufferSize,
                m_llPacketCounter,
                m_ulLastOsWritePacket
            ));
        }''')

        s = replace_once(
            s,
            '        WriteBytes(ByteDisplacement);',
            position + '        WriteBytes(ByteDisplacement);'
        )

        s = replace_once(
            s,
            '        ReadBytes(ByteDisplacement);',
            position + render_position + '        ReadBytes(ByteDisplacement);'
        )
        s = replace_once(s, '    _In_ LARGE_INTEGER ilQPC\n)', '    _In_ LARGE_INTEGER ilQPC,\n    _In_ ULONG diagnosticOrigin\n)')
        if s.count('UpdatePosition(ilQPC);') != 2:
            raise ValueError('Expected GetPosition and GetPacketCount update sites')
        s = s.replace('UpdatePosition(ilQPC);', 'UpdatePosition(ilQPC, 1);', 1)
        s = s.replace('UpdatePosition(ilQPC);', 'UpdatePosition(ilQPC, 2);', 1)
        s = replace_once(s, '_this->UpdatePosition(qpc);', '_this->UpdatePosition(qpc, 3);')
        s = replace_once(s, '    BOOL bufferCompleted = FALSE;', '    BOOL bufferCompleted = FALSE;' + debug('    ULONG diagnosticSignals = 0;'))
        s = replace_once(s, '            KeSetEvent(nleCurrent->NotificationEvent, 0, 0);',
            '            KeSetEvent(nleCurrent->NotificationEvent, 0, 0);' + debug('            ++diagnosticSignals;'))
        s = replace_once(s, 'End:\n    KeReleaseSpinLock(&_this->m_PositionSpinLock, oldIrql);', 'End:' + debug('''    _this->m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Notification(_this->m_bCapture,
        qpc.QuadPart, TimeElapsedInMS, bufferCompleted, _this->m_hnsDPCTimeCarryForward,
        _this->m_ullLinearPosition, _this->m_llPacketCounter, diagnosticSignals,
        _this->m_ulCurrentWritePosition, static_cast<ULONG>(_this->m_ullWritePosition),
        _this->m_ulDmaBufferSize, _this->m_ulNotificationsPerBuffer);''') + '    KeReleaseSpinLock(&_this->m_PositionSpinLock, oldIrql);')
        def timeline(code):
            return '\n#if VIRMIXER_SHARED_TIMELINE\n' + code + '\n#endif\n'
        s = replace_once(s, '    UNREFERENCED_PARAMETER(diagnosticOrigin);',
            timeline('    const auto cableToken = m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->BeginTransfer(m_bCapture);') + '    UNREFERENCED_PARAMETER(diagnosticOrigin);')
        s = replace_once(s, '        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Reset();', '''    {
#if VIRMIXER_SHARED_TIMELINE
        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->TimelineTransition(m_bCapture);
#else
        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Reset();
#endif
    }''')
        s = replace_once(s, anchor, anchor + timeline('''            m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->TimelineRun(m_bCapture,
                ullPerfCounterTemp.QuadPart, m_ullPerformanceCounterFrequency.QuadPart, m_ullLinearPosition);'''))
        for helper in ('ReadBytes', 'WriteBytes'):
            s = replace_once(s, helper+'(ByteDisplacement);', helper+'(ByteDisplacement\n#if VIRMIXER_SHARED_TIMELINE\n            , cableToken\n#endif\n        );')
        s = s.replace('    _In_ ULONG ByteDisplacement\n)', '    _In_ ULONG ByteDisplacement\n#if VIRMIXER_SHARED_TIMELINE\n    , VirtualCable::TransferToken cableToken\n#endif\n)')
        prefix = '    const ULONG skipped = ByteDisplacement > m_ulDmaBufferSize ? ByteDisplacement - m_ulDmaBufferSize : 0;'
        s = s.replace(prefix, timeline('''    if (!m_bCapture) m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->ExpireRender(
        cableToken, m_ullLinearPosition, ByteDisplacement, m_ulDmaBufferSize);''') + prefix + timeline('    ULONGLONG cableLinear = m_ullLinearPosition + skipped;'))
        old_read = '''        const ULONG cableActual =
                    m_pMiniport->GetAdapterCommObj()
                        ->GetVirtualCable()
                        ->Read(m_pDmaBuffer + bufferOffset, runWrite);'''
        s = replace_once(s, old_read, '''#if VIRMIXER_SHARED_TIMELINE
        const ULONG cableActual = m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->ReadAt(
            cableToken, cableLinear, m_pDmaBuffer + bufferOffset, runWrite);
#else
''' + old_read + '\n#endif')
        old_write = '        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->Write(m_pDmaBuffer + bufferOffset, runWrite);'
        s = replace_once(s, old_write, '''#if VIRMIXER_SHARED_TIMELINE
        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->WriteAt(
            cableToken, cableLinear, m_pDmaBuffer + bufferOffset, runWrite);
#else
''' + old_write + '\n#endif')
        s = s.replace('        ByteDisplacement -= runWrite;', '        ByteDisplacement -= runWrite;' + timeline('        cableLinear += runWrite;'))
        # Diagnostic mode only. Keep the existing RENDERPOS path for normal B,
        # but avoid callback printing while the bounded identity probe is active.
        s = s.replace('        if (m_ulNotificationsPerBuffer > 0)\n        {\n            const LONG dmaRead',
                      '        if (!VIRMIXER_FRAME_PROBE && m_ulNotificationsPerBuffer > 0)\n        {\n            const LONG dmaRead')
        probe_context = '''
#if VIRMIXER_FRAME_PROBE
        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->ProbeContext(m_bCapture,
            {static_cast<ULONGLONG>(ilQPC.QuadPart), m_ullLinearPosition,
             static_cast<ULONGLONG>(m_llPacketCounter), ByteDisplacement,
             m_ulDmaBufferSize, m_ulCurrentWritePosition, m_ulLastOsWritePacket, diagnosticOrigin});
#endif
'''
        for helper, capture in (('ReadBytes', False), ('WriteBytes', True)):
            call = helper + '(ByteDisplacement\n#if VIRMIXER_SHARED_TIMELINE\n            , cableToken\n#endif\n        );'
            observe = '''
#if VIRMIXER_FRAME_PROBE
        m_pMiniport->GetAdapterCommObj()->GetVirtualCable()->ProbeDma(''' + str(capture).lower() + ''',
            cableToken, m_ullLinearPosition, ByteDisplacement, m_pDmaBuffer, m_ulDmaBufferSize);
#endif
'''
            s = replace_once(s, '        ' + call,
                             probe_context + ('        ' + call + observe if capture else observe + '        ' + call))
        return (s.replace('Write sine wave to buffer.', 'Copy cable PCM (or silence) to capture buffer.')
                .replace('This function writes the audio buffer using a sine wave generator', 'This function fills capture DMA from the PCM cable, padding underruns with silence.')
                .replace('This function reads the audio buffer and saves the data in a file.', 'This function routes render DMA into the PCM cable.'))
    edit('EndpointsCommon/minwavertstream.cpp', lambda s: format_stream_indentation(stream(s)))
    edit('EndpointsCommon/minwavertstream.h', lambda s: replace_once(s,
        '        _In_ LARGE_INTEGER ilQPC\n', '        _In_ LARGE_INTEGER ilQPC,\n        _In_ ULONG diagnosticOrigin = 4\n')
        .replace('#include "savedata.h"', '#include "savedata.h"\n#include "VirtualCable.h"')
        .replace('        _In_ ULONG ByteDisplacement\n', '        _In_ ULONG ByteDisplacement\n#if VIRMIXER_SHARED_TIMELINE\n        , VirtualCable::TransferToken cableToken\n#endif\n'))
    edit('EndpointsCommon/MiniportAudioEngineNode.cpp', guard_optional_sideband)

    def pairs(s):
        s = array_body(s, 'g_RenderEndpoints', '    &SpdifMiniports,')
        s = array_body(s, 'g_CaptureEndpoints', '    &MicInMiniports,')
        # No hardware offloading or extra render/capture devices in this derivative.
        return s.replace('ENDPOINT_OFFLOAD_SUPPORTED,', '0, // VirMixer: no offload')
    edit('TabletAudioSample/minipairs.h', pairs)

    def formats(s, capture=False):
        names = re.findall(r'KSDATAFORMAT_WAVEFORMATEXTENSIBLE\s+(\w+)\[\]', s)
        for name in names:
            s = array_body(s, name, FORMAT)
            s = re.sub(re.escape(name) + r'\[(?:\d+|SIZEOF_ARRAY\(' + re.escape(name) + r'\)\s*-\s*1)\]', name + '[0]', s)
        s = re.sub(r'(#define\s+\w*(?:MIN|MAX)_SAMPLE_RATE\s+)\d+', r'\g<1>48000', s)
        if capture:
            s = re.sub(r'(#define\s+MICIN_DEVICE_MAX_CHANNELS\s+)\d+', r'\g<1>2', s)
            s = re.sub(r'(#define\s+MICIN_MAX_INPUT_STREAMS\s+)\d+', r'\g<1>1', s)
        else:
            for name, value in [('SPDIF_MAX_INPUT_SYSTEM_STREAMS', 1), ('SPDIF_MAX_INPUT_OFFLOAD_STREAMS', 0), ('SPDIF_MAX_OUTPUT_LOOPBACK_STREAMS', 0)]:
                s = re.sub(r'(#define\s+' + name + r'\s+)\w+', r'\g<1>' + str(value), s)
            # Do not advertise compressed AC3 passthrough in a PCM-only cable.
            s = s.replace('STATICGUIDOF(KSDATAFORMAT_SUBTYPE_DOLBY_AC3_SPDIF)', 'STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM)')
            s = s.replace('STATICGUIDOF(KSDATAFORMAT_SUBTYPE_IEC61937_DOLBY_DIGITAL)', 'STATICGUIDOF(KSDATAFORMAT_SUBTYPE_PCM)')
            s = array_body(s, 'SpdifPinDataRangePointersStream',
                '    PKSDATARANGE(&SpdifPinDataRangesStream[0]),\n    PKSDATARANGE(&PinDataRangeAttributeList)')
        return s
    edit('TabletAudioSample/micinwavtable.h', lambda s: formats(s, True))
    edit('TabletAudioSample/spdifwavtable.h', formats)
    # Do not advertise sample hardware volume/peak nodes that do not process PCM.
    edit('TabletAudioSample/micintoptable.h', lambda s: array_body(s,
        'MicInMiniportConnections', '    { PCFILTER_NODE, KSPIN_TOPO_MIC_ELEMENTS, PCFILTER_NODE, KSPIN_TOPO_BRIDGE }')
        .replace('SIZEOF_ARRAY(MicInTopologyNodes),', '0,').replace('  MicInTopologyNodes,', '  NULL,')
        .replace('KSAUDIO_SPEAKER_MONO', 'KSAUDIO_SPEAKER_STEREO')
        .replace('0xd48deb08, 0xfd1c, 0x4d1e, 0xb8, 0x21, 0x90, 0x64, 0xd4, 0x9a, 0xe9, 0x6e',
                 '0xf63c1cf1, 0x5e27, 0x4c45, 0x9a, 0xb3, 0xa6, 0x12, 0x4a, 0xb8, 0xf1, 0x77'))
    edit('TabletAudioSample/spdiftoptable.h', lambda s: s.replace(
        '//=============================================================================',
        'static const GUID VirMixerRenderName = {0x058cf48c, 0x8958, 0x4b7e, {0x96, 0x14, 0xb3, 0xa1, 0x0e, 0x9d, 0x66, 0xb7}};\n\n//=============================================================================', 1)
        .replace('NULL,                                             // Name', '&VirMixerRenderName,                              // Name'))
    # Disable sideband services and kernel recording in this dedicated device.
    edit('adapter.cpp', lambda s: re.sub(r'(DPF\(D_VERBOSE, \("DoNotCreateDataFiles:.*?;)',
        r'\1\n    g_DoNotCreateDataFiles = 1; // VirMixer never records in kernel mode.', s))
    for relative in ('TabletAudioSample/TabletAudioSample.vcxproj', 'EndpointsCommon/EndpointsCommon.vcxproj'):
        def project(s, relative=relative):
            # Also remove a final token without a trailing semicolon (Release configs).
            s = re.sub(r'\b(?:SYSVAD_BTH_BYPASS|SYSVAD_USB_SIDEBAND|SYSVAD_A2DP_SIDEBAND)\b;?', '', s)
            if relative.startswith('Tablet'):
                s = s.replace('<Inf Exclude="@(Inx)" Include="*.inx" />', '<Inf Include="VirMixerAudio.inx" />')
                if s.count('<TargetName>TabletAudioSample</TargetName>') != 4:
                    raise ValueError('Expected four upstream TargetName configuration overrides')
                s = s.replace('<TargetName>TabletAudioSample</TargetName>', '<TargetName>VirMixerAudio</TargetName>')
            # Both translation-unit groups must agree on VirtualCable layout.
            # Release has neither trace storage nor trace calls.
            pattern = re.compile(
                r'(<ItemDefinitionGroup Condition="\'\$\(Configuration\)\|\$\(Platform\)\'==\'Debug\|x64\'">.*?'
                r'<ClCompile>.*?<PreprocessorDefinitions>)'
                r'([^<]*)'
                r'(</PreprocessorDefinitions>)', re.S)
            s, replacements = pattern.subn(r'\1\2;VIRMIXER_DIAGNOSTICS=1;VIRMIXER_FRAME_PROBE=' + str(int(args.frame_probe)) + r'\3', s, count=1)
            if replacements != 1:
                raise ValueError('Expected Debug x64 compiler definitions: ' + relative)
            s = s.replace('<PreprocessorDefinitions>%(PreprocessorDefinitions);',
                '<PreprocessorDefinitions>%(PreprocessorDefinitions);VIRMIXER_SHARED_TIMELINE=' + str(int(args.shared_timeline)) + ';')
            return s
        edit(relative, project)
    shutil.copy2(ROOT / 'package' / 'VirMixerAudio.inx', base / 'TabletAudioSample' / 'VirMixerAudio.inx')
    (destination / 'UPSTREAM.txt').write_text(f'Microsoft Windows-driver-samples\n{REVISION}\n', encoding='utf-8')
    paths = [Path('audio/sysvad') / p.relative_to(upstream / 'audio/sysvad')
             for p in (upstream / 'audio/sysvad').rglob('*') if p.is_file()]
    paths += [Path(p) for p in ['LICENSE-Microsoft', 'UPSTREAM.txt', 'audio/sysvad/AudioRing.h',
              'audio/sysvad/VirtualCable.h', 'audio/sysvad/StreamTrace.h', 'audio/sysvad/CableTimeline.h', 'audio/sysvad/FrameProbe.h', 'audio/sysvad/TabletAudioSample/VirMixerAudio.inx']]
    manifest_path.write_text(json.dumps({p.as_posix(): hashlib.sha256((destination / p).read_bytes()).hexdigest()
                                        for p in paths}, indent=2), encoding='utf-8')
    (destination / '.virmixer-mode.json').write_text(json.dumps({'mode': 'B' if args.shared_timeline else 'A', 'frameProbe': args.frame_probe}), encoding='utf-8')
    print(f'Generated source: {base}\nGeneration only; compile with driver/build.ps1. No driver installed.')


if __name__ == '__main__':
    main()
