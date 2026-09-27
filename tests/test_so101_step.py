import copy
import unittest

from harness.safety import SafetyConfig
from harness.so101_step import NAMES, execute_step, validate, hold_measured


class FakeHardware:
    def __init__(self, mode='reached'):
        self.now = 0.0
        self.mode = mode
        self.writes = []
        self.events = []
        self.failed_once = False
        self.state = {reg: dict.fromkeys(NAMES, value) for reg, value in (
            ('Present_Position', 2000), ('Goal_Position', 2000), ('Present_Load', 0),
            ('Torque_Enable', 1), ('Status', 0), ('Present_Temperature', 30), ('Present_Voltage', 120))}

    def read(self):
        self.now += 0.01
        active = self.writes and self.writes[-1][1] == 'requested_bounded_step'
        if active:
            if self.mode == 'disconnected' or (self.mode == 'one_read_failure' and not self.failed_once):
                self.failed_once = True
                raise OSError('fake serial failure')
            if self.mode == 'stalled':
                self.state['Present_Position']['shoulder_lift'] = 2003
            elif self.mode == 'rebound':
                self.state['Present_Position']['shoulder_lift'] = 2020 if self.now < 0.3 else 2003
            else:
                self.state['Present_Position']['shoulder_lift'] = self.state['Goal_Position']['shoulder_lift']
            if self.mode == 'load':
                self.state['Present_Load']['shoulder_lift'] = -84
            if self.mode == 'brief_load':
                self.state['Present_Load']['shoulder_lift'] = -84 if self.now < 0.18 else 0
            if self.mode == 'peak_load':
                self.state['Present_Load']['shoulder_lift'] = -125
            if self.mode == 'hot':
                self.state['Present_Temperature']['elbow_flex'] = 46
        return copy.deepcopy(self.state)

    def write(self, goal, reason):
        self.writes.append((goal, reason))
        self.state['Goal_Position']['shoulder_lift'] = goal

    def sleep(self, seconds):
        self.now += seconds

    def emit(self, event, payload):
        self.events.append((event, copy.deepcopy(payload)))


class MonitoredStepTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {n: {'range_min': 1000, 'range_max': 3000} for n in NAMES}
        self.config = SafetyConfig('fake', 3.0, 20, 150, dict.fromkeys(NAMES, (1000, 3000)))

    def run_step(self, mode):
        fake = FakeHardware(mode)
        result = execute_step(fake.read, fake.write, fake.emit, self.calibration, self.config,
                              'shoulder_lift', 20, clock=lambda: fake.now, sleep=fake.sleep)
        return fake, result

    def test_reached_requires_final_observations(self):
        fake, result = self.run_step('reached')
        self.assertTrue(result['success'])
        self.assertEqual(result['outcome'], 'target_reached')
        self.assertEqual(result['actual_delta'], 20)
        self.assertEqual(fake.writes, [(2020, 'requested_bounded_step')])

    def test_stall_stops_early_and_latches_actual_without_false_success(self):
        fake, result = self.run_step('stalled')
        self.assertFalse(result['success'])
        self.assertEqual(result['outcome'], 'stalled')
        self.assertLess(fake.now, 1.5)
        self.assertEqual(fake.writes[-1], (2003, 'stop_at_measured_position'))

    def test_passing_through_target_then_rebounding_is_failure(self):
        _, result = self.run_step('rebound')
        self.assertFalse(result['success'])
        self.assertEqual(result['outcome'], 'stalled')

    def test_raw_load_and_temperature_stop_independently_of_status(self):
        for mode in ('load', 'hot'):
            with self.subTest(mode=mode):
                fake, result = self.run_step(mode)
                self.assertFalse(result['success'])
                self.assertEqual(result['outcome'], 'guard_stopped')
                self.assertEqual(fake.writes[-1][1], 'stop_at_measured_position')

    def test_transient_read_failure_attempts_only_measured_hold(self):
        fake, result = self.run_step('one_read_failure')
        self.assertFalse(result['success'])
        self.assertEqual(result['outcome'], 'error')
        self.assertEqual(len(fake.writes), 2)
        self.assertEqual(fake.writes[-1][1], 'stop_at_measured_position')

    def test_short_load_transient_is_distinct_from_persistent_load(self):
        _, short = self.run_step('brief_load')
        self.assertTrue(short['success'])
        fake, persistent = self.run_step('load')
        self.assertFalse(persistent['success'])
        self.assertEqual(persistent['outcome'], 'guard_stopped')
        self.assertLess(fake.now, 0.7)

    def test_peak_load_stops_on_first_sample(self):
        fake, result = self.run_step('peak_load')
        self.assertFalse(result['success'])
        self.assertEqual(len([e for e, _ in fake.events if e == 'sample']), 1)

    def test_persistent_read_failure_does_not_invent_a_recovery_position(self):
        fake, result = self.run_step('disconnected')
        self.assertFalse(result['success'])
        self.assertIn('recovery_error', result)
        self.assertEqual(fake.writes, [(2020, 'requested_bounded_step')])

    def test_precondition_failure_never_writes(self):
        for register, value in (('Status', 1), ('Torque_Enable', 0),
                                ('Present_Load', 108), ('Present_Position', 3100)):
            with self.subTest(register=register):
                fake = FakeHardware()
                fake.state[register]['elbow_flex'] = value
                with self.assertRaises(ValueError):
                    execute_step(fake.read, fake.write, fake.emit, self.calibration, self.config,
                                 'shoulder_lift', 20, clock=lambda: fake.now, sleep=fake.sleep)
                self.assertEqual(fake.writes, [])

    def test_step_limit_applies_to_actual_and_existing_goal(self):
        state = FakeHardware().state
        for delta in (0, 41, -41, 2.5):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                validate(state, self.calibration, self.config, 'shoulder_lift', delta)
        state['Goal_Position']['shoulder_lift'] = 1970
        with self.assertRaises(ValueError):
            validate(state, self.calibration, self.config, 'shoulder_lift', 20)

    def test_existing_gravity_error_is_preserved_when_abandoning_a_target(self):
        class GravityHardware(FakeHardware):
            def read(self):
                self.now += 0.01
                if self.writes:
                    offset = 9 if self.writes[-1][1] == 'requested_bounded_step' else 8
                    self.state['Present_Position']['shoulder_lift'] = self.state['Goal_Position']['shoulder_lift'] + offset
                return copy.deepcopy(self.state)

        fake = GravityHardware()
        fake.state['Goal_Position']['shoulder_lift'] = 1992
        result = execute_step(fake.read, fake.write, fake.emit, self.calibration, self.config,
                              'shoulder_lift', 10, clock=lambda: fake.now, sleep=fake.sleep)
        self.assertFalse(result['success'])  # Overshoot remains a failed step.
        self.assertEqual(result['recovery_hold_bias_counts'], -8)
        self.assertEqual(fake.writes[-1], (2011, 'stop_at_measured_position_with_existing_bias'))
        # The old zero-error hold would set 2019 and sag to 2027.
        self.assertEqual(result['after']['Present_Position']['shoulder_lift'], 2019)
        self.assertTrue(result['recovery_state_within_limits'])

    def test_unsettled_gravity_reference_does_not_write(self):
        fake = FakeHardware()
        fake.state['Goal_Position']['shoulder_lift'] = 1994
        def moving_read():
            state = fake.read()
            if fake.now > 0.1:
                state['Present_Position']['shoulder_lift'] += 3
            return state
        with self.assertRaisesRegex(ValueError, 'has not settled'):
            execute_step(moving_read, fake.write, fake.emit, self.calibration, self.config,
                         'shoulder_lift', 10, clock=lambda: fake.now, sleep=fake.sleep)
        self.assertEqual(fake.writes, [])

    def test_biased_hold_checks_calibration_and_original_envelope(self):
        for actual, bias, origin, joint in ((2037, 8, 2000, 'shoulder_lift'),
                                          (1003, -8, 1010, 'shoulder_lift'),
                                          (2000, 9, 2000, 'shoulder_lift'),
                                          (2000, 6, 2000, 'gripper')):
            with self.subTest(actual=actual, bias=bias, joint=joint):
                fake = FakeHardware()
                fake.state['Present_Position'][joint] = actual
                fake.state['Goal_Position'][joint] = actual
                with self.assertRaises(ValueError):
                    hold_measured(fake.read, fake.write, self.calibration, joint,
                                  bias=bias, origin=origin)
                self.assertEqual(fake.writes, [])


if __name__ == '__main__':
    unittest.main()
