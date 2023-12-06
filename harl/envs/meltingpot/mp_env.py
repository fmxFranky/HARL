import numpy as np
from meltingpot import scenario, substrate

from . import utils
from .multiagentenv import MultiAgentEnv


class MeltingPotEnv(MultiAgentEnv):
    """MeltingPotEnv is a wrapper around a substrate and scenario.

    It provides a MultiAgentEnv interface to the substrate and scenario.
    """

    def __init__(self, scenario_name, downsample_scale=1):
        if scenario_name[-1].isdigit():
            env = scenario.get_factory(scenario_name).build()
        else:
            substrate_name = scenario_name[:-2]
            env = substrate.build(
                substrate_name,
                roles=substrate.get_config(substrate_name).default_player_roles,
            )
        self.n_agents = self.num_agents = len(env.observation_spec())

        env = utils.DownSamplingSubstrateWrapper(env, downsample_scale)
        self.env = env
        observation_space = utils.spec_to_space(env.observation_spec())
        self.observation_space = [obs_space["RGB"] for obs_space in observation_space]
        self.share_observation_space = [
            obs_space["WORLD.RGB"] for obs_space in observation_space
        ]
        self.action_space = utils.spec_to_space(env.action_spec())

    def reset(self):
        timestep = self.env.reset()
        obs = []
        state = []
        available_actions = np.ones((self.num_agents, self.action_space[0].n))
        for observation in timestep.observation:
            obs.append(observation["RGB"])
            state.append(observation["WORLD.RGB"])
        return np.array(obs), np.array(state), available_actions

    def step(self, actions):
        # input()
        timestep = self.env.step(np.array(actions, dtype=np.int32).flatten())
        obs = []
        state = []
        reward = []
        done = [timestep.last()] * self.num_agents
        available_actions = np.ones((self.num_agents, self.action_space[0].n))
        for observation in timestep.observation:
            obs.append(observation["RGB"])
            state.append(observation["WORLD.RGB"])
            reward.append(observation["COLLECTIVE_REWARD"])
        return (
            np.array(obs),
            np.array(state),
            np.array(reward)[:, None],
            done,
            [{}] * self.num_agents,
            available_actions,
        )

    def get_avail_actions(self):
        return [1] * self.num_agents

    def seed(self, seed):
        pass

    def get_num_agents(self):
        return self.num_agents

    def get_spaces(self):
        return self.observation_space, self.share_observation_space, self.action_space
