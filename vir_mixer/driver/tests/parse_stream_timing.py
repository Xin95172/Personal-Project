"""Read deferred Debug stream snapshots; never infer a root cause from a flag alone."""
import argparse
import json
import re
from pathlib import Path

FIELDS = {
    'TRACE': 'seq epoch kind side from to qpc W Req Actual prefill req avail take prefillBytes shortageBytes',
    'CLOCK': 'seq side id state run freq qpc prev now carryIn carryOut',
    'POSITION': 'seq side linear disp dma interval notifications origin eos',
    'TOTAL': 'seq side updates disp copied skipped epochDisp epochCopied timerQpc timerDelta timers timerCarry maxDelta maxQpc',
    'EPOCH': 'seq side runHns runLinear runCarry posEpoch transferEpoch crossEpoch',
    'NOTIFY': 'seq side qpc linear count events packet elapsed gate carry signals crossed osWrite dmaWrite',
}


def parse(text):
    groups, errors, ends = {}, [], []
    for line in text.splitlines():
        match = re.search(r'VirMixer: (TRACE_END|TRACE|CLOCK|POSITION|TOTAL|EPOCH|NOTIFY) (.*)', line)
        if not match:
            continue
        kind, body = match.groups()
        values = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', body)}
        if kind == 'TRACE_END':
            ends.append(values.get('dropped', -1))
            continue
        if not set(FIELDS[kind].split()) <= values.keys():
            errors.append(f'incomplete {kind} seq={values.get("seq")}')
            continue
        seq = values['seq']
        key = kind if kind == 'TRACE' else f'{kind}{values["side"]}'
        group = groups.setdefault(seq, {})
        if key in group:
            errors.append(f'duplicate {key} seq={seq}; use one adapter/session per log')
        group[key] = values
    required = {'TRACE'} | {f'{kind}{side}' for kind in FIELDS if kind != 'TRACE' for side in (0, 1)}
    complete = []
    for seq, group in sorted(groups.items()):
        if not required <= group.keys():
            errors.append(f'missing snapshot parts seq={seq}')
        else:
            complete.append(group)
    if groups and not ends:
        errors.append('missing TRACE_END; drain/log may be incomplete')
    if any(n != 0 for n in ends):
        errors.append('trace capacity exceeded or invalid drain marker; do not claim complete history')
    sequences = sorted(groups)
    if any(b != a + 1 for a, b in zip(sequences, sequences[1:])):
        errors.append('sequence gaps; do not claim complete history')
    return complete, errors, ends


def evidence(group):
    e = group['TRACE']
    result = {
        'seq': e['seq'], 'epoch': e['epoch'], 'kind': e['kind'],
        'request_lead_ms': (e['Req'] - e['W']) / 192,
        'prefill_ms': e['prefillBytes'] / 192,
        'shortage_ms': e['shortageBytes'] / 192,
        'request_minus_actual_matches_silence': e['Req'] - e['Actual'] == e['prefillBytes'] + e['shortageBytes'],
        'actual_exceeds_written': e['Actual'] > e['W'],
        'streams': [],
    }
    clocks = [group[f'CLOCK{i}'] for i in (0, 1)]
    if clocks[0]['freq'] and clocks[0]['freq'] == clocks[1]['freq']:
        result['capture_minus_render_run_ms'] = (clocks[1]['run'] - clocks[0]['run']) * 1000 / clocks[0]['freq']
        if all(group[f'TOTAL{i}']['updates'] for i in (0, 1)):
            result['capture_minus_render_last_update_ms'] = (clocks[1]['qpc'] - clocks[0]['qpc']) * 1000 / clocks[0]['freq']
    for i, c in enumerate(clocks):
        p, t = group[f'POSITION{i}'], group[f'TOTAL{i}']
        epoch, notify = group[f'EPOCH{i}'], group[f'NOTIFY{i}']
        item = {'side': i, 'state': c['state'], 'origin': p['origin'], 'updates': t['updates'],
                'displacement': p['disp'], 'dma_bytes': p['dma'], 'interval_ms': p['interval'],
                'cumulative_displacement': t['disp'], 'cumulative_copied': t['copied'],
                'cumulative_dma_skipped': t['skipped'],
                'epoch_displacement': t['epochDisp'], 'epoch_copied': t['epochCopied'],
                'transfers_across_reset': epoch['crossEpoch'],
                'position_epoch': epoch['posEpoch'], 'transfer_epoch': epoch['transferEpoch'],
                'notification': notify}
        if c['freq'] and t['timers']:
            item['first_timer_since_run_ms'] = (t['timerQpc'] - c['run']) * 1000 / c['freq'] if t['timers'] == 1 else None
        if notify['qpc'] >= c['run'] and notify['qpc'] and p['notifications'] and p['dma']:
            packet_bytes = p['dma'] // p['notifications']
            if packet_bytes:
                item['notification_packet_bytes'] = packet_bytes
                item['notification_period_exact_ms'] = packet_bytes / 192
                item['notification_count_minus_dma_packets'] = notify['count'] - (notify['linear'] - epoch['runLinear']) // packet_bytes
                item['signaled_without_one_boundary'] = bool(notify['signals'] and notify['crossed'] != 1 and not p['eos'])
        if c['freq'] and t['updates']:
            elapsed = c['now'] - c['prev'] + c['carryIn']
            # Matches pinned source's cast-before-division; long (>429s) gaps must be flagged.
            expected = ((elapsed & 0xffffffff) // 10000) * 192
            item.update(update_elapsed_ms=(c['now'] - c['prev']) / 10000,
                        update_age_ms=(e['qpc'] - c['qpc']) * 1000 / c['freq'],
                        timer_delta_ms=t['timerDelta'] * 1000 / c['freq'],
                        max_timer_delta_ms=t['maxDelta'] * 1000 / c['freq'],
                        first_update_since_run_ms=(c['qpc'] - c['run']) * 1000 / c['freq'] if t['updates'] == 1 else None,
                        expected_displacement_before_eos=expected,
                        displacement_matches_qpc=None if p['eos'] else p['disp'] == expected,
                        carry_matches_qpc=c['carryOut'] == elapsed % 10000,
                        elapsed_exceeds_ulong=elapsed > 0xffffffff,
                        current_dma_skip=max(0, p['disp'] - p['dma']),
                        cumulative_expected_before_eos=((c['now'] - epoch['runHns'] + epoch['runCarry']) // 10000) * 192)
            item['cumulative_displacement_residual'] = t['disp'] - item['cumulative_expected_before_eos']
        result['streams'].append(item)
    return result


def summarize(text):
    groups, errors, ends = parse(text)
    return {'complete_snapshots': len(groups), 'integrity_errors': errors, 'drains_dropped': ends,
            'events': [evidence(g) for g in groups],
            'root_cause': 'UNPROVEN: compare state/reset history, prefill, paired update ages, displacement and DMA skips. '
                          'Snapshots occur before transfer or partway through a wrapped transfer; copied totals may lag pending displacement.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = args.log.read_bytes()
    text = data.decode('utf-16') if data.startswith((b'\xff\xfe', b'\xfe\xff')) else data.decode('utf-8-sig', errors='replace')
    result = json.dumps(summarize(text), indent=2)
    if args.output:
        args.output.write_text(result + '\n', encoding='utf-8')
    else:
        print(result)
