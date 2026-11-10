import numpy as np
import pytest
from pettingzoo.test import api_test

from catan.agents.baselines import RandomAgent
from catan.engine import RulesConfig
from catan.env import CatanAECEnv, CatanSingleAgentEnv


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pettingzoo_api():
    api_test(CatanAECEnv(config=RulesConfig(max_turns_per_game=60)), num_cycles=2000)


def test_aec_full_game_with_masks():
    env = CatanAECEnv()
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        legal = np.flatnonzero(obs["action_mask"])
        env.step(int(rng.choice(legal)))
    assert env.agents == []


def test_aec_rewards_sum_to_winner_signal():
    env = CatanAECEnv(loss_reward=-1 / 3)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    total = dict.fromkeys(env.possible_agents, 0.0)
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        total[agent] += reward
        env.step(None if term or trunc else int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    assert sum(total.values()) == pytest.approx(0.0, abs=1e-9)
    assert max(total.values()) == 1.0


def test_single_agent_env_runs_to_completion():
    env = CatanSingleAgentEnv(opponents=lambda: [RandomAgent() for _ in range(3)])
    obs, info = env.reset(seed=11)
    assert obs.shape == env.observation_space.shape
    rng = np.random.default_rng(0)
    done, steps = False, 0
    while not done:
        assert info["action_mask"][...].any()
        a = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        steps += 1
    assert "winner" in info and steps > 10
