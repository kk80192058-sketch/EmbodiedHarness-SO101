import unittest

from harness.session_analysis import pair_completed_steps


def row(event, marker):
    return {'event': event, 'payload': {'marker': marker}}


class SessionAnalysisTests(unittest.TestCase):
    def test_pairs_each_write_with_its_own_before_result_and_after(self):
        rows = [
            row('observation', 'idle'),
            row('observation', 'before-a'), row('write_intent', 'intent-a'),
            row('result', 'result-a'), row('observation', 'after-a'),
            row('observation', 'before-b'), row('write_intent', 'intent-b'),
            row('result', 'result-b'), row('observation', 'after-b'),
        ]
        pairs = pair_completed_steps(rows)
        self.assertEqual([(p['before']['marker'], p['intent']['marker'], p['result']['marker'], p['after']['marker'])
                          for p in pairs],
                         [('before-a', 'intent-a', 'result-a', 'after-a'),
                          ('before-b', 'intent-b', 'result-b', 'after-b')])

    def test_incomplete_or_superseded_intents_cannot_create_a_sample(self):
        rows = [
            row('observation', 'before-a'), row('write_intent', 'intent-a'),
            row('write_intent', 'intent-b'), row('result', 'result-b'),
            row('observation', 'after-b'), row('write_intent', 'intent-c'),
        ]
        pairs = pair_completed_steps(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['intent']['marker'], 'intent-b')
