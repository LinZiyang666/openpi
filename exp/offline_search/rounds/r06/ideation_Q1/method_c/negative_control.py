"""Reverse control for replay_test's §7b checks (code test only).

Patches the pre-7b rule (slow_ambiguous forces p=0) into the deployed controller
and requires replay_test to reject it. Output goes to a scratch probe directory.
"""
import sys
from . import methods, replay_test


def main():
    methods.call_probability = lambda state, cooled, nominal: (0. if cooled or state == 'slow_ambiguous'
                                                              else 1. if state == 'slow_confirmed' else nominal)
    sys.argv = ['negative_control', '--cell', 'pi05_l10_50', '--case', 'stall',
                '--out', '/tmp/q1_method_c_fits/probe_7b/negative_control_old_rule']
    try:
        replay_test.main()
    except AssertionError as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f'PASS: replay_test rejected the pre-7b controller at replay_test.py:{tb.lineno}: {tb.line}')
    else:
        raise SystemExit('FAIL: pre-7b controller passed the 7b replay checks')


if __name__ == '__main__':
    main()
