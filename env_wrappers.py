import time
import numpy as np
import torch
from gym import spaces
from inspect import getfullargspec

class GymWrapper(object):
    """
    for multi-agent
    """
    def __init__(self, env):
        # print("gym init", getfullargspec(env.reset).args)
        # import traceback
        # for line in traceback.format_stack():
        #    print(line.strip())
        self.env = env

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''
        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else: # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    # success means turn on the tj curriculum
    def reset(self, epoch, success=False):
        reset_args = getfullargspec(self.env.reset).args
        # print("reset args", reset_args, self.env.reset)
        if self.env.name == 'TrafficJunction':
            # print("env_wrapper.py", epoch, success)
            obs = self.env.reset(epoch=epoch, success=success)
        elif 'epoch' in reset_args:
            #print(epoch, "reset epoch good")
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        obs = self._flatten_obs(obs)
        return obs

    def display(self):
        self.env.render()
        time.sleep(0.5)

    def end_display(self):
        self.env.exit_render()

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this

        if self.dim_actions == 1:
            # this basically makes sure you are ignoring the gating action.
            action = action[0]

        obs, r, done, info = self.env.step(action)
        obs = self._flatten_obs(obs)

        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        obs = obs.reshape(1, -1, self.observation_dim)
        obs = torch.from_numpy(obs).double()
        return obs

    def get_stat(self):
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()

    def get_loc_wrapper(self):
        return self.env.get_loc()

    def get_reached_wrapper(self):
        return self.env.get_is_completed()

    def get_alive_wrapper(self,i):
        return self.env.get_alive(i)

    def get_path_wrapper(self, i):
        return self.env.get_path(i)

    def obs2pos_wrapper(self,obs):
        return self.env.obs2pos(obs)

    def get_vocab_size_wrapper(self):
        return self.env.get_vocab_size()


class StarcraftWrapper(object):
    """
    for multi-agent
    """
    def __init__(self, env):
        # print("gym init", getfullargspec(env.reset).args)
        # import traceback
        # for line in traceback.format_stack():
        #    print(line.strip())
        self.env = env
        self.env_info = env.get_env_info()
        self.n_actions = self.env_info["n_actions"]
        self.n_agents = self.env_info["n_agents"]

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''
        return self.env.get_obs_size()

    @property
    def num_actions(self):
        return self.env.get_total_actions()

    @property
    def dim_actions(self):
        # Discrete => only 1 action takes place at a time.
        return 1

    @property
    def action_space(self):
        return spaces.Discrete(self.env.get_total_actions())

    def reset(self, epoch, success=False):
        obs, full_state = self.env.reset()
        obs = np.array(obs)
        obs = self._flatten_obs(obs)
        return obs

    def display(self):
        raise NotImplementedError

    def end_display(self):
        raise NotImplementedError

    def step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this

        if self.dim_actions == 1:
            # this basically makes sure you are ignoring the gating action.
            action = action[0]
        for i in range(len(action)):
            if not self.env.get_avail_agent_actions(i)[action[i]]:
                if self.env.get_avail_agent_actions(i)[0]:
                    # agent is dead, take no-op
                    action[i] = 0
                else:
                    action[i] = 1 # stop by default when invalid action
        # print(action)
        # print(self.env.get_avail_actions())
        r, done, info = self.env.step(action)
        r = np.ones(self.n_agents) * r
        obs = np.array(self.env.get_obs())
        obs = self._flatten_obs(obs)

        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        obs = obs.reshape(1, -1, self.observation_dim)
        obs = torch.from_numpy(obs).double()
        return obs

    def get_stat(self):
        return self.env.get_stats()
