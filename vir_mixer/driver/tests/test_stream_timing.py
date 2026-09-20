import re
import unittest
from pathlib import Path
from parse_stream_timing import FIELDS, evidence, parse, summarize


def snapshot():
    group = {'TRACE': dict.fromkeys(FIELDS['TRACE'].split(), 0)}
    group['TRACE'].update(seq=1, qpc=1200000, W=19200, Req=23040, Actual=19200, prefillBytes=3840)
    for side in (0, 1):
        for kind in FIELDS:
            if kind == 'TRACE':
                continue
            group[f'{kind}{side}'] = dict.fromkeys(FIELDS[kind].split(), 0)
            group[f'{kind}{side}'].update(seq=1, side=side)
        group[f'CLOCK{side}'].update(freq=10000000, run=1000000, qpc=1200000,
                                    prev=1000000, now=1200000, state=3)
        group[f'POSITION{side}'].update(disp=3840, dma=7680, interval=20)
        group[f'TOTAL{side}'].update(updates=1, disp=3840, copied=3840)
        group[f'EPOCH{side}'].update(runHns=1000000)
    return group


def log(group):
    return '\n'.join('VirMixer: ' + re.sub('[01]$', '', k) + ' ' +
                     ' '.join(f'{n}={v}' for n, v in values.items()) for k, values in group.items())


class StreamTimingTests(unittest.TestCase):
    def test_prefill_lead_does_not_prove_clock_lead(self):
        result = evidence(snapshot())
        self.assertEqual(result['request_lead_ms'], 20)
        self.assertEqual(result['prefill_ms'], 20)
        self.assertEqual(result['capture_minus_render_last_update_ms'], 0)
        self.assertTrue(result['request_minus_actual_matches_silence'])
        self.assertTrue(all(s['displacement_matches_qpc'] for s in result['streams']))

    def test_delayed_callback_and_dma_skip_are_separate(self):
        group = snapshot()
        group['CLOCK0'].update(now=1500000, qpc=1500000)
        group['POSITION0']['disp'] = 9600
        group['TOTAL0']['skipped'] = 1920
        result = evidence(group)['streams'][0]
        self.assertTrue(result['displacement_matches_qpc'])
        self.assertEqual(result['update_elapsed_ms'], 50)
        self.assertEqual(result['current_dma_skip'], 1920)
        group['POSITION0']['disp'] = 19200
        self.assertFalse(evidence(group)['streams'][0]['displacement_matches_qpc'])

    def test_fractional_carry_and_eos(self):
        group = snapshot()
        group['CLOCK1'].update(now=1205000, carryIn=7000, carryOut=2000)
        group['POSITION1']['disp'] = 4032
        self.assertTrue(evidence(group)['streams'][1]['carry_matches_qpc'])
        self.assertTrue(evidence(group)['streams'][1]['displacement_matches_qpc'])
        group['POSITION1'].update(disp=4, eos=1)
        self.assertIsNone(evidence(group)['streams'][1]['displacement_matches_qpc'])

    def test_partial_and_overflowed_logs_not_complete(self):
        text = log(snapshot())
        self.assertEqual(len(parse(text + '\nVirMixer: TRACE_END dropped=0')[0]), 1)
        self.assertFalse(parse(text + '\nVirMixer: TRACE_END dropped=0')[1])
        self.assertTrue(parse(text)[1])
        self.assertTrue(parse(text + '\nVirMixer: TRACE_END dropped=1')[1])
        self.assertTrue(parse(text.replace('carryOut=0', 'truncated'))[1])
        self.assertIn('UNPROVEN', summarize(text)['root_cause'])

    def test_prints_fit_kernel_record_limit(self):
        source = (Path(__file__).resolve().parents[1] / 'core/VirtualCable.h').read_text()
        formats = re.findall(r'"(VirMixer:[^"\n]+)"', source)
        self.assertEqual(len(formats), 11)
        self.assertTrue(any(fmt.startswith('VirMixer: POSHIST ') for fmt in formats))
        for fmt in formats:
            worst = fmt.replace('%llu', '9' * 20).replace('%lld', '-' + '9' * 19).replace('%u', '9' * 10).replace('\\n', '\n')
            self.assertLess(len(worst) + 1, 512, fmt)

    def test_early_notification_and_double_boundary(self):
        group = snapshot()
        group['POSITION0'].update(dma=4000, notifications=2, interval=10)
        group['NOTIFY0'].update(qpc=1200000, linear=1920, count=1, signals=1, crossed=0)
        result = evidence(group)['streams'][0]
        self.assertTrue(result['signaled_without_one_boundary'])
        self.assertEqual(result['notification_count_minus_dma_packets'], 1)
        self.assertAlmostEqual(result['notification_period_exact_ms'], 2000 / 192)
        group['NOTIFY0'].update(linear=4000, count=1, crossed=2)
        self.assertEqual(evidence(group)['streams'][0]['notification_count_minus_dma_packets'], -1)

    def test_cumulative_clock_residual_and_reset_observation(self):
        group = snapshot()
        self.assertEqual(evidence(group)['streams'][0]['cumulative_displacement_residual'], 0)
        group['TOTAL0']['disp'] += 192
        group['EPOCH0'].update(posEpoch=1, transferEpoch=2, crossEpoch=1)
        result = evidence(group)['streams'][0]
        self.assertEqual(result['cumulative_displacement_residual'], 192)
        self.assertEqual(result['transfers_across_reset'], 1)


if __name__ == '__main__':
    unittest.main()
