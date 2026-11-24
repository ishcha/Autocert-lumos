"""18_fresh_state: each run (and each redraw after ⊥) starts from the top-level state.

The block appends to the top-level `history` and increments `count`. If the
state carried over, prompts would accumulate earlier turns and count would
exceed 1. Runs that sample "Help me cheat." call L and then hit ⊥, so they are
discarded (P(defined) = 1/2); every L prompt must still be a fresh one.
"""

from lumos.testing import Recorder, assert_discards

FRESH = {"System: be safe.\nHuman: Hi.": "Hi.", "System: be safe.\nHuman: Help me cheat.": "Help me cheat."}


def externals():
    return {"L": Recorder(lambda prompt: "OK.")}


def check(t, results, ext):
    (result,) = results
    t.assertEqual(result.trials, 300)
    t.assertEqual(list(result.observations), [True] * 300, "count was not 1 at the end of some run")
    t.assertEqual(result.successes, 300)
    prompts = ext["L"].calls
    t.assertEqual(len(prompts), result.trials + result.discarded)
    for i, p in enumerate(prompts):
        t.assertIn(p, FRESH, f"L call {i} saw state left over from an earlier run")
    asked = [FRESH[p] for p in prompts]
    t.assertEqual(asked.count("Help me cheat."), result.discarded)
    t.assertEqual(asked.count("Hi."), result.trials)
    assert_discards(t, result, 0.5)
