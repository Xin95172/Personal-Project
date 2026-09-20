"""Summarize bounded stall runs without declaring the historical bug solved."""
import argparse
import hashlib
import json
from pathlib import Path
from frame_identity import analyze_log
from parse_stream_timing import parse


def analyze(folder):
    cases = json.loads((folder / 'matrix.json').read_text(encoding='utf-8'))
    kernel = analyze_log(folder / 'kernel.log')
    snapshots, errors, ends = parse((folder / 'kernel.log').read_text(encoding='utf-8-sig'))
    rows = []
    for case in cases:
        report = case.get('report')
        if report is None:
            report = json.loads((folder / (case['name'] + '.json')).read_text())
        stalls = report['stalls']
        intervals = [(s['begin_qpc'], s['end_qpc'] + report['qpc_frequency'] // 20) for s in stalls]
        clamps = []
        for snapshot in snapshots:
            trace = snapshot['TRACE']
            if trace['kind'] != 7 or not any(a <= trace['qpc'] <= b for a, b in intervals):
                continue
            side = trace['side']
            pos = snapshot[f'POSITION{side}']
            clamps.append(dict(side=side, qpc=trace['qpc'], linear=pos['linear'],
                               displacement=pos['disp'], dma=pos['dma'], skipped_frames=(pos['disp']-pos['dma'])//4))
        packets = report.get('wasapi_packets', {})
        gaps = [p for p in packets.get('records', []) if p.get('position_gap')]
        rows.append(dict(name=case['name'], passed=report['passed'], capture=report['capture'],
                         identity_usable=report['capture']['invalid_frames'] == 0,
                         stalls=stalls, clamps_near_stall=clamps, packet_position_gaps=gaps,
                         wasapi_discontinuity_packets=packets.get('discontinuity_packets'),
                         input_overflows=report['input_overflows'], output_underflows=report['output_underflows'],
                         errors=report['errors']))
    return dict(cases=rows, frame_drains=kernel['summaries'], frame_log_complete=kernel['complete'],
                frame_anomalies=kernel['anomalies'], stream_integrity_errors=errors, stream_drains=ends,
                kernel_sha256=hashlib.sha256((folder / 'kernel.log').read_bytes()).hexdigest(),
                scope='Injected stalls localize a failure mechanism; not proof of the old spontaneous +48/+352 root cause.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = analyze(args.folder)
    (args.folder / 'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('frame complete:', result['frame_log_complete'], 'stream integrity:', result['stream_integrity_errors'])
    for row in result['cases']:
        print(row['name'], 'source gaps:', row['capture']['source_gap_frames'] if row['identity_usable'] else 'INVALID ENCODING; do not interpret as frame loss',
              'packet gaps:', [g['position_gap'] for g in row['packet_position_gaps']],
              'clamps:', [(c['side'], c['skipped_frames']) for c in row['clamps_near_stall']])
