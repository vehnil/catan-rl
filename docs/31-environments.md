# RL environments

`src/catan/env/` wraps the engine in standard RL interfaces. Both environments emit the 1,388-float feature vector from `player/features.py` plus the 488-entry legal-action mask.

## `CatanAECEnv`: PettingZoo AEC (multi-agent)
Passes `pettingzoo.test.api_test`.

- Agents are `player_0` … `player_3`. `agent_selection` follows the engine's `actor`, so out-of-turn discards and trade responses come up naturally.
- Observation: `{"observation": float32[1388], "action_mask": int8[488]}` (PettingZoo classic-games convention).
- Rewards: `win_reward` (+1) to the winner and `loss_reward` (−1) to the others at the end; set `loss_reward=-1/3` for zero-sum. Optional per-step shaping: `vp_reward_scale × ΔVP`. Hitting the turn cap sets `truncations`, not `terminations`.

```python
env = CatanAECEnv()
env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = None if term or trunc else my_policy(obs["observation"], obs["action_mask"])
    env.step(action)
```

## `CatanSingleAgentEnv`: Gymnasium (one learner)
The learner sits in one seat (random each episode unless `learner_seat` is set). `opponents` is a factory returning 3 `Agent`s, which act inside `step()` until it's the learner's move again. This is the one place the env layer touches the agent layer.

The mask is in `info["action_mask"]` and `action_masks()` (sb3-contrib MaskablePPO convention), so off-the-shelf maskable algorithms work.

## Tests

`tests/env/test_envs.py`: PettingZoo `api_test`, a full AEC game with masked actions, terminal rewards vs. winner, and a full single-agent episode.
