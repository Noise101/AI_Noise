#!/usr/bin/env python3
"""Ablation: full temporal search versus a near-immediate-only learner."""

from temporal_agent_v2 import simulate


def rate(results, key):
    return sum(bool(result[key]) for result in results) / len(results)


if __name__ == "__main__":
    trials = 200
    temporal = [simulate(seed, max_lag=7) for seed in range(trials)]
    myopic = [simulate(seed, max_lag=1) for seed in range(trials)]
    print("agent                 before-change  after-change")
    print(f"temporal search       {rate(temporal, 'before'):>12.1%}  {rate(temporal, 'after'):>12.1%}")
    print(f"lag-1-only ablation   {rate(myopic, 'before'):>12.1%}  {rate(myopic, 'after'):>12.1%}")
