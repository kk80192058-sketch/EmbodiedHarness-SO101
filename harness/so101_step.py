"""Hardware-independent supervision of one already-authorized SO-101 step.

The load threshold below is a conservative session stop threshold in raw
register units, not a manufacturer's overload rating. No torque/gain writes.
"""
from dataclasses import replace

from harness.safety import SafetyConfig, SafetyGate

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
REGISTERS = ('Torque_Enable', 'Present_Position', 'Goal_Position', 'Present_Load',
             'Present_Voltage', 'Present_Temperature', 'Status')
LOAD_STOP_RAW = 75
LOAD_PEAK_STOP_RAW = 125
LOAD_MAX_ELEVATED_S = 0.2
TEMPERATURE_STOP = 45
POSITION_TOLERANCE = 8
GRAVITY_JOINTS = ('shoulder_lift', 'elbow_flex')
MAX_EXISTING_HOLD_BIAS = 8


def state_fault(state, calibration, *, load_limit=LOAD_STOP_RAW):
    for name in NAMES:
        pos = state['Present_Position'][name]
        if not calibration[name]['range_min'] <= pos <= calibration[name]['range_max']:
            return f'{name}: outside calibration'
        if state['Torque_Enable'][name] != 1:
            return f'{name}: torque is not enabled'
        if state['Status'][name] != 0:
            return f'{name}: nonzero status'
        if state['Present_Temperature'][name] >= TEMPERATURE_STOP:
            return f'{name}: temperature stop'
        if abs(state['Present_Load'][name]) >= load_limit:
            return f'{name}: raw load stop'
    return None


def validate(state, calibration, config: SafetyConfig, joint, delta):
    if joint not in NAMES or type(delta) is not int or not 0 < abs(delta) <= 40:
        raise ValueError('One known joint and an integer step of 1..40 counts are required')
    fault = state_fault(state, calibration)
    if fault:
        raise ValueError(fault)
    current = state['Present_Position']
    goal = SafetyGate(replace(config, max_step_counts=40)).validate_raw_joint_goals(
        current, {joint: current[joint] + delta}, timeout_s=min(2.5, config.action_timeout_s))[joint]
    if abs(goal - state['Goal_Position'][joint]) > 40:
        raise ValueError('New goal differs from existing goal by more than 40')
    return goal


def hold_measured(read, write, calibration, joint, *, bias=0, origin=None):
    """Fresh read, validate, latch only the selected joint, and read back.

    Recovery may run with elevated load, but must not latch a disconnected,
    unpowered, faulted, out-of-range or unexpectedly distant motor.
    """
    if type(bias) is not int or abs(bias) > MAX_EXISTING_HOLD_BIAS:
        raise ValueError('Unverified holding bias')
    if bias and joint not in GRAVITY_JOINTS:
        raise ValueError('Holding bias is only supported on gravity-bearing joints')
    state = read()
    actual = state['Present_Position'][joint]
    goal = actual + bias
    entry = calibration[joint]
    if not all(entry['range_min'] <= value <= entry['range_max'] for value in (actual, goal)):
        raise ValueError('Cannot hold an out-of-calibration position')
    if state['Torque_Enable'][joint] != 1 or state['Status'][joint] != 0:
        raise ValueError('Cannot hold: selected motor torque/status gate failed')
    if abs(goal - state['Goal_Position'][joint]) > 40:
        raise ValueError('Cannot hold: goal change exceeds 40 counts')
    if origin is not None and abs(goal - origin) > 40:
        raise ValueError('Cannot hold: recovery goal exceeds the original 40-count envelope')
    write(goal, 'stop_at_measured_position_with_existing_bias' if bias else 'stop_at_measured_position')
    return read()


def execute_step(read, write, emit, calibration, config, joint, delta, *, clock, sleep):
    """Injectable I/O keeps fake-bus failure tests entirely away from hardware.

    write receives only a goal for the single joint chosen by the caller.
    emit must persist its event before returning. A successful receipt requires
    several final readings near the original target, with no safety stop.
    """
    before = read()
    hold_bias = 0
    old_bias = before['Goal_Position'][joint] - before['Present_Position'][joint] if joint in GRAVITY_JOINTS else 0
    if 0 < abs(old_bias) <= MAX_EXISTING_HOLD_BIAS:
        # A proportional position controller can need a nonzero error to hold
        # against gravity. Zeroing that existing error caused another sag in
        # the 12:09 trace. Reuse only a small, freshly observed stable offset;
        # this is not a new gain, integral term, or estimated gravity model.
        emit('preflight_hold_reference', before)
        sleep(0.12)
        confirmed = read()
        if (abs(confirmed['Present_Position'][joint] - before['Present_Position'][joint]) > 1
                or confirmed['Goal_Position'][joint] != before['Goal_Position'][joint]):
            raise ValueError('Gravity-bearing joint has not settled before the step')
        before = confirmed
        candidate = before['Goal_Position'][joint] - before['Present_Position'][joint]
        if abs(candidate) <= MAX_EXISTING_HOLD_BIAS:
            hold_bias = candidate
    emit('before', before)
    target = validate(before, calibration, config, joint, delta)
    result = {'success': False, 'outcome': 'error', 'target': target, 'before': before,
              'recovery_hold_bias_counts': hold_bias}
    attempted = False
    try:
        attempted = True  # A failed serial write can still have reached the motor.
        write(target, 'requested_bounded_step')
        start = clock()
        budget = min(2.0, config.action_timeout_s)
        samples = []
        stop = None
        elevated_since = None
        while clock() - start < budget:
            state = read()
            elapsed = clock() - start
            samples.append((elapsed, state['Present_Position'][joint]))
            emit('sample', {'elapsed_s': elapsed, **state})
            fault = state_fault(state, calibration, load_limit=LOAD_PEAK_STOP_RAW)
            if fault:
                stop = 'guard_stopped'
                result['stop_reason'] = fault
                break
            elevated = any(abs(v) >= LOAD_STOP_RAW for v in state['Present_Load'].values())
            if elevated:
                if elevated_since is None:
                    elevated_since = elapsed
                if elapsed - elevated_since >= LOAD_MAX_ELEVATED_S:
                    stop = 'guard_stopped'
                    result['stop_reason'] = 'Raw load remained elevated for 0.2 seconds'
                    break
            else:
                elevated_since = None
            for name in NAMES:
                if abs(state['Present_Position'][name] - before['Present_Position'][name]) > 40:
                    stop = 'guard_stopped'
                    result['stop_reason'] = f'{name}: observed travel exceeds 40 counts'
                    break
            if stop:
                break
            recent = [(t, pos) for t, pos in samples if t >= elapsed - 0.6]
            if (len(recent) >= 4 and recent[-1][0] - recent[0][0] >= 0.45
                    and max(pos for _, pos in recent) - min(pos for _, pos in recent) <= 2
                    and abs(samples[-1][1] - target) > POSITION_TOLERANCE):
                stop = 'stalled'
                result['stop_reason'] = 'Position plateaued outside target tolerance'
                break
            sleep(0.08)
        final = read()
        emit('pre_hold_final', final)
        fault = state_fault(final, calibration)
        if fault:
            stop = 'guard_stopped'
            result['stop_reason'] = fault
        settled = (len(samples) >= 3 and samples[-1][0] - samples[-3][0] >= 0.12
                   and all(abs(pos - target) <= POSITION_TOLERANCE for _, pos in samples[-3:])
                   and abs(final['Present_Position'][joint] - target) <= POSITION_TOLERANCE)
        result['outcome'] = stop or ('target_reached' if settled else 'target_not_reached')
        result['success'] = result['outcome'] == 'target_reached'
        if not result['success']:
            final = hold_measured(read, write, calibration, joint, bias=hold_bias,
                                  origin=before['Present_Position'][joint])
            sleep(0.25)
            final = read()
            emit('held_final', final)
            recovery_fault = state_fault(final, calibration)
            if any(abs(final['Present_Position'][name] - before['Present_Position'][name]) > 40 for name in NAMES):
                recovery_fault = 'Observed travel exceeds 40 counts after recovery'
            result['recovery_state_within_limits'] = recovery_fault is None
            if recovery_fault:
                result['recovery_error'] = recovery_fault
        result['after'] = final
        result['actual_delta'] = final['Present_Position'][joint] - before['Present_Position'][joint]
        result['target_error'] = final['Present_Position'][joint] - target
    except BaseException as error:
        result['success'] = False
        result['outcome'] = 'error'
        result['error'] = repr(error)
        if attempted:
            try:
                result['after'] = hold_measured(read, write, calibration, joint, bias=hold_bias,
                                                origin=before['Present_Position'][joint])
                emit('error_recovery', result['after'])
            except BaseException as recovery_error:
                result['recovery_error'] = repr(recovery_error)
        # A communication failure cannot prove either holding or torque-off.
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            emit('result', result)
            raise
    emit('result', result)
    return result
