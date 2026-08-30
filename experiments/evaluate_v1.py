#!/usr/bin/env python3
"""Evaluate v1 across seeds instead of trusting one attractive trace."""

from __future__ import annotations

import argparse
import random
import statistics

from causal_agent_v1 import ChangingWorld, RuleInducer


def trial(seed: int, steps: int, change_at: int) -> dict[str, float | bool]:
    world = ChangingWorld(change_at=change_at)
    agent = RuleInducer(world.sensor_count, random.Random(seed))
    rewards = 0
    recovery_step = steps
    consecutive_correct = 0
    before_correct = False

    for step in range(steps):
        observation, _ = agent.choose(world.candidates())
        reward = world.reward(observation, step)
        rewards += reward
        agent.learn(observation, reward)
        learned = str(agent.best_rule()[0])
        if step == change_at - 1:
            before_correct = learned == "s0 & !s3"
        if step >= change_at and learned == "s2 | s5":
            consecutive_correct += 1
            if consecutive_correct == 3:
                recovery_step = step - change_at - 2
        else:
            consecutive_correct = 0

    return {
        "before_correct": before_correct,
        "after_correct": str(agent.best_rule()[0]) == "s2 | s5",
        "recovery": recovery_step,
        "reward_rate": rewards / steps,
    }


def main(trials: int, steps: int, change_at: int) -> None:
    results = [trial(seed, steps, change_at) for seed in range(trials)]
    recovered = [float(r["recovery"]) for r in results if r["recovery"] < steps]
    print(f"trials={trials}, steps={steps}, change_at={change_at}")
    print(f"rule before change: {sum(bool(r['before_correct']) for r in results) / trials:.1%}")
    print(f"rule after change:  {sum(bool(r['after_correct']) for r in results) / trials:.1%}")
    print(f"recovered:          {len(recovered) / trials:.1%}")
    print(f"median recovery:    {statistics.median(recovered) if recovered else float('inf'):.1f} steps")
    print(f"mean reward rate:   {statistics.mean(float(r['reward_rate']) for r in results):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--change-at", type=int, default=35)
    args = parser.parse_args()
    main(args.trials, args.steps, args.change_at)
