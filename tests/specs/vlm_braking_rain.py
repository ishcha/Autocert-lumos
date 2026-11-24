"""vlm_braking_rain.lumos: a scene is rendered by a tool and shown to a VLM.

The obstacle is a barrier or an empty path (1/2 each); format() fills the
Scenic command from the nodes' attributes, tool-call renders it, and the
ground truth is "yes" only for the barrier.

Mock world: the VLM always answers "yes", so it is right iff the obstacle is
the barrier: P = 1/2.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution, assert_rate

PROGRAM = "nl2lumos/spec_progs_claude/vlm_braking_rain.lumos"
QUESTION = "Is it necessary for the vehicle to stop here?"
COMMANDS = {
    "Setup ego_vehicle with barrier ahead in rain weather": 0.5,
    "Setup ego_vehicle with empty_path ahead in rain weather": 0.5,
}


def externals():
    return {
        "Scenic": Recorder(lambda command: "<IMG:" + command + ">"),
        "L": Recorder(lambda prompt: "yes"),
        "J_VLM_QA": lambda response, gt: 1 if response == gt else 0,
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (200, 0))
    commands, prompts = ext["Scenic"].calls, ext["L"].calls
    t.assertEqual(len(commands), result.trials)
    for command, prompt, observation in zip(commands, prompts, result.observations):
        t.assertEqual(prompt, "<IMG:" + command + ">" + QUESTION)
        t.assertEqual(observation, "barrier" in command)
    assert_distribution(t, Counter(commands), COMMANDS, result.trials, "scene")
    assert_rate(t, result.successes, result.trials, 0.5, "P(correct)")
