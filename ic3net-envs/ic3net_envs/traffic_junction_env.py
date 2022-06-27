#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Each agent can observe itself (it's own identity) i.e. s_j = j and vision, path ahead of it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

# core modules
import random
import math
import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from ic3net_envs.traffic_helper import *
from inspect import getargspec

def nPr(n,r):
    f = math.factorial
    return f(n)//f(n-r)

class TrafficJunctionEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"
        self.name  = "TrafficJunction"
        #print("init traffic junction", getargspec(self.reset).args)
        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10

        self.episode_over = False
        self.has_failed = 0
        self.collision_location = []
        self.pos2enc = None
        self.enc2pos = None
        self.is_toy_ex = False

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Traffic Junction task')
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box (i.e length of road) ")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of car")
        env.add_argument('--add_rate_min', type=float, default=0.05,
                         help="rate at which to add car (till curr. start)")
        env.add_argument('--add_rate_max', type=float, default=0.2,
                         help=" max rate at which to add car")
        # env.add_argument('--curr_start', type=float, default=0,
        #                  help="start making harder after this many epochs [0]")
        # env.add_argument('--curr_end', type=float, default=0,
        #                  help="when to make the game hardest [0]")
        env.add_argument('--difficulty', type=str, default='easy',
                         help="Difficulty level, easy|medium|hard|longer_easy")
        env.add_argument('--vocab_type', type=str, default='bool',
                         help="Type of location vector to use, bool|scalar")

        # updated curriculum parameters
        env.add_argument('--curr_start_epoch', type=float, default=-1.,
                         help="start making harder after this many epochs [0]")
        env.add_argument('--curr_epochs', type=float, default=1000.,
                         help="Number of epochs of curriculum for when to make the game hardest [0]")



    def multi_agent_init(self, args):
        #print("init tj")
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'add_rate_min', 'add_rate_max', 'curr_start_epoch', 'curr_epochs',
                  'difficulty', 'vocab_type']

        for key in params:
            setattr(self, key, getattr(args, key))

        self.difficulty = args.difficulty
        self.ncar = args.nagents
        self.dims = dims = (self.dim, self.dim)
        difficulty = args.difficulty
        vision = args.vision

        if difficulty in ['medium','easy','longer_easy']:
            assert dims[0]%2 == 0, 'Only even dimension supported for now.'

            assert dims[0] >= 4 + vision, 'Min dim: 4 + vision'

        if difficulty == 'hard':
            assert dims[0] >= 9, 'Min dim: 9'
            assert dims[0]%3 ==0, 'Hard version works for multiple of 3. dim. only.'

        # Add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do -
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.naction = 2
        self.action_space = spaces.Discrete(self.naction)

        # make no. of dims odd for easy case.
        if difficulty == 'easy' or difficulty == 'longer_easy':
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        nroad = {'easy':2,
                'medium':4,
                'hard':8,
                'longer_easy':6}

        dim_sum = dims[0] + dims[1]
        base = {'easy':   dim_sum,
                'medium': 2 * dim_sum,
                'hard':   4 * dim_sum,
                'longer_easy': dim_sum}

        self.npath = nPr(nroad[difficulty],2)

        # Setting max vocab size for 1-hot encoding
        if self.vocab_type == 'bool':
            self.BASE = base[difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            # car_type + base + outside + 0-index
            self.vocab_size = 1 + self.BASE + 1 + 1
            self.observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
            # self.observation_space = spaces.Tuple((
            #                         spaces.Discrete(self.naction),
            #                         spaces.Discrete(self.npath),
            #                         spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size+1))))#+1 on vocab size for extra get_alive obs
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, len(dims), vision * vision * vocab)
            self.observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiDiscrete(dims),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
            # Actual observation will be of the shape 1 * ncar * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        self._set_grid()

        if difficulty == 'easy' or difficulty == 'longer_easy':
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

        return

    def reset(self, epoch=None, success=False):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.ncar)
        self.wait = np.zeros(self.ncar)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [0] * self.ncar
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.ncar

        # self.cars = np.zeros(self.ncar)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS,self.CAR_CLASS + self.ncar)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.ncar, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.ncar, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.ncar, - 1)



        if "special" in epoch:
            epoch_ = epoch.split("_")
            epoch = int(epoch_[1])
            # if self.difficulty != "easy":
            #     print ("toy env only implemented for easy environment.")
            #     assert False
            #
            # self.is_toy_ex = True
            # agent_0_idx = 0
            # agent_1_idx = 1
            #
            # self.alive_mask[agent_0_idx] = 1
            # self.alive_mask[agent_1_idx] = 1
            #
            #
            # #assign agent 0 to route 0 and agent 1 to route 1
            # agent_0_p_i = 0
            # agent_1_p_i = 1
            #
            # self.route_id[agent_0_idx] = agent_0_p_i
            # self.route_id[agent_1_idx] = agent_1_p_i
            #
            # self.chosen_path[agent_0_idx] = self.routes[agent_0_p_i][0]
            #
            # self.chosen_path[agent_1_idx] = self.routes[agent_1_p_i][0]
            #
            # #
            # # self.car_route_loc[agent_0_idx] = 3-int(epoch_[2])-1
            # # self.car_route_loc[agent_1_idx] = 3-int(epoch_[2])-1
            # #
            # # self.car_loc[agent_0_idx] = self.routes[agent_0_p_i][0][3-int(epoch_[2])]
            # # self.car_loc[agent_1_idx] = self.routes[agent_1_p_i][0][3-int(epoch_[2])]
            #
            # self.car_route_loc[agent_0_idx] = 0
            # self.car_route_loc[agent_1_idx] = 0
            #
            # self.car_loc[agent_0_idx] = self.routes[agent_0_p_i][0][0]
            # self.car_loc[agent_1_idx] = self.routes[agent_1_p_i][0][0]
            #
            # self.cars_in_sys += 2

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = self.curr_epochs
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        # print("reached first step", epoch, epoch_range, add_rate_range, self.epoch_last_update)
        if success and self.curr_start_epoch == -1:
            self.curr_start_epoch = epoch
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update and self.curr_start_epoch != -1:
            # print("running curriculum now")
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be ncar * vision * vision ndarray
        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either ncar or ncar x 1

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (ncar x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over.
            info (dict) : diagnostic information useful for debugging.
        """

        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either ncar or ncar x 1
        action = np.array(action).squeeze()
        # print ("taking action: " + str(action))

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        assert len(action) == self.ncar, "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.ncar)

        for i, a in enumerate(action):
            self._take_action(i, a)
        # print ("took actions: " + str(self.car_last_act))

        self._add_cars()

        obs = self._get_obs()
        reward = self._get_reward()

        debug = {'car_loc':self.car_loc,
                'alive_mask': np.copy(self.alive_mask),
                'wait': self.wait,
                'cars_in_sys': self.cars_in_sys,
                'is_completed': np.copy(self.is_completed)}

        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate

        return obs, reward, self.episode_over, debug

    def render(self, mode='human', close=False):

        grid = self.grid.copy().astype(object)
        # grid = np.zeros(self.dims[0]*self.dims[1], dtypeobject).reshape(self.dims)
        grid[grid != self.OUTSIDE_CLASS] = '_'
        grid[grid == self.OUTSIDE_CLASS] = ''
        self.stdscr.clear()
        for i, p in enumerate(self.car_loc):
            if self.car_last_act[i] == 0: # GAS
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<>'
                else:
                    grid[p[0]][p[1]] = '<>'
            else: # BRAKE
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<b>'
                else:
                    grid[p[0]][p[1]] = '<b>'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if row_num == idx == 0:
                    continue
                if item != '_':
                    if '<>' in item and len(item) > 3: #CRASH, one car accelerates
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2))
                    elif '<>' in item: #GAS
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    elif 'b' in item and len(item) > 3: #CRASH
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2))
                    elif 'b' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(5))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '_'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self,i):
        np.random.seed(i)
        return

    def _set_grid(self):
        self.grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty)
        for road in roads:
            self.grid[road] = self.ROAD_CLASS

        if self.vocab_type == 'bool':
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(self.grid[road].shape)
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_obs(self):
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1


        # remove the outside class.
        if self.vocab_type == 'scalar':
            self.bool_base_grid = self.bool_base_grid[:,:,1:]

        # print ("registered last actions:" + str(self.car_last_act / (self.naction - 1)))
        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.naction - 1)


            # route id
            r_i = self.route_id[i] / (self.npath - 1)

            # loc
            p_norm = p / (h-1, w-1)

            # vision square
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = self.bool_base_grid[slice_y, slice_x]

            # alive_vec = [self.get_alive(i) for i in range(len(self.car_loc))]


            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)

            # v_sq = np.expand_dims(np.expand_dims(np.insert(v_sq[0][0],-1,1-self.get_alive(i)),axis=0),axis=0)

            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = tuple(obs)

        return obs


    def _add_cars(self):
        if self.is_toy_ex:
            return

        for r_i, routes in enumerate(self.routes):
            if self.cars_in_sys >= self.ncar:
                return

            # Add car to system and set on path
            if np.random.uniform() <= self.add_rate:

                # chose dead car on random
                idx = self._choose_dead()
                # make it alive
                self.alive_mask[idx] = 1

                # choose path randomly & set it
                p_i = np.random.choice(len(routes))
                # make sure all self.routes have equal len/ same no. of routes
                self.route_id[idx] = p_i + r_i * len(routes)
                self.chosen_path[idx] = routes[p_i]

                # set its start loc
                self.car_route_loc[idx] = 0
                self.car_loc[idx] = routes[p_i][0]

                # increase count
                self.cars_in_sys += 1

    def _set_paths_easy(self):
        h, w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': []
        }

        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        self.routes = list(self.routes.values())


    def _set_paths_medium_old(self):
        h,w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': [],
            'RIGHT': [],
            'DOWN': []
        }

        # type 0 paths: go straight on junction
        # type 1 paths: take right on junction
        # type 2 paths: take left on junction


        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2-1) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to UP to LEFT, type 1
        first_half = full[:h//2]
        second_half = [(h//2 - 1, i) for i in range(w//2 - 2,-1,-1) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))

        # 2 refers to UP to RIGHT, type 2
        second_half = [(h//2, i) for i in range(w//2-1, w) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))


        # 3 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        # 4 refers to LEFT to DOWN, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2 - 1) for i in range(h//2+1, h)]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))

        # 5 refers to LEFT to UP, type 2
        second_half = [(i, w//2) for i in range(h//2, -1,-1) ]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))


        # 6 refers to DOWN to UP, type 0
        full = [(i, w//2) for i in range(h-1,-1,-1)]
        self.routes['DOWN'].append(np.array([*full]))

        # 7 refers to DOWN to RIGHT, type 1
        first_half = full[:h//2]
        second_half = [(h//2, i) for i in range(w//2+1,w)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))

        # 8 refers to DOWN to LEFT, type 2
        second_half = [(h//2-1, i) for i in range(w//2,-1,-1)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))


        # 9 refers to RIGHT to LEFT, type 0
        full = [(h//2-1, i) for i in range(w-1,-1,-1)]
        self.routes['RIGHT'].append(np.array([*full]))

        # 10 refers to RIGHT to UP, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2) for i in range(h//2 -2, -1,-1)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))

        # 11 refers to RIGHT to DOWN, type 2
        second_half = [(i, w//2-1) for i in range(h//2-1, h)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))


        # PATHS_i: 0 to 11
        # 0 refers to UP to down,
        # 1 refers to UP to left,
        # 2 refers to UP to right,
        # 3 refers to LEFT to right,
        # 4 refers to LEFT to down,
        # 5 refers to LEFT to up,
        # 6 refers to DOWN to up,
        # 7 refers to DOWN to right,
        # 8 refers to DOWN to left,
        # 9 refers to RIGHT to left,
        # 10 refers to RIGHT to up,
        # 11 refers to RIGHT to down,

        # Convert to routes dict to list of paths
        paths = []
        for r in self.routes.values():
            for p in r:
                paths.append(p)

        # Check number of paths
        # assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)

    def _set_paths(self, difficulty):
        route_grid = self.route_grid if self.vocab_type == 'bool' else self.grid
        self.routes = get_routes(self.dims, route_grid, difficulty)

        # Convert/unroll routes which is a list of list of paths
        paths = []
        for r in self.routes:
            for p in r:
                paths.append(p)

        # Check number of paths
        assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)


    def _unittest_path(self,paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis =1)
            if np.any(step_jump != 1):
                print("Any", p, i)
                return False
            if not np.all(step_jump == 1):
                print("All", p, i)
                return False
        return True


    def _take_action(self, idx, act):
        # non-active car
        if self.alive_mask[idx] == 0:
            return

        # add wait time for active cars
        self.wait[idx] += 1

        # action BRAKE i.e STAY
        if act == 1:
            self.car_last_act[idx] = 1
            return

        # GAS or move
        if act==0:
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]

            # car/agent has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
                self.is_completed[idx] = 1
                return

            elif curr > len(self.chosen_path[idx]):
                print(curr)
                raise RuntimeError("Out of boud car path")

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 0



    def _get_reward(self):
        reward = np.full(self.ncar, self.TIMESTEP_PENALTY) * self.wait

        for i, l in enumerate(self.car_loc):
            if (len(np.where(np.all(self.car_loc[:i] == l,axis=1))[0]) or \
               len(np.where(np.all(self.car_loc[i+1:] == l,axis=1))[0])) and l.any():
               reward[i] += self.CRASH_PENALTY
               self.collision_location = l.tolist()
               self.has_failed = 1

        reward = self.alive_mask * reward
        return reward

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1 # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])


    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_epochs)
        mod_val = int(step_size / step)

        # if self.curr_start <= epoch < self.curr_end and (epoch - self.curr_start) % mod_val == 0:
        if self.curr_start_epoch <= epoch < self.curr_start_epoch+self.curr_epochs and (epoch - self.curr_start_epoch) % mod_val == 0:
            self.exact_rate = self.exact_rate + step_size
            self.add_rate = self.exact_rate
            print("tj curriculum", self.add_rate)
            # self.add_rate = step_size * (self.exact_rate // step_size)
        else:
            print("not updating curriculum for tj for epoch", epoch)

    def get_loc(self):
        return self.car_loc

    def list2tup(self,list):
        return (*list, )

    def get_grid(self):
        grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty)
        for road in roads:
            grid[road] = self.ROAD_CLASS
        return grid

    def init_pos2enc(self):
        # w, h = self.dims
        # temp_loc = self.car_loc.copy()
        # temp_act = self.car_last_act.copy()
        # tempr_ri =  self.route_id.copy()
        # alive_mask = self.alive_mask.copy()
        #
        # print (self.car_loc)
        # possible_pos = []
        # for i in range(h):
        #     for j in range(w):
        #         possible_pos.append([i,j])
        #
        # self.car_loc = np.array(possible_pos)
        # self.car_last_act = np.zeros(len(possible_pos))
        # self.route_id = np.zeros(len(possible_pos))
        # self.alive_mask = np.ones(len(possible_pos))
        #
        # o = np.array(self._get_obs())[:,2]
        # print (o)
        # self.pos2enc = {}
        # self.enc2pos = {}
        #
        # for pos, enc_pos in zip(possible_pos, o):
        #     enc_pos = enc_pos[0][0]
        #     self.pos2enc[tuple(pos)] = tuple(enc_pos[:-1])
        #     self.enc2pos[tuple(enc_pos[:-1])] = tuple(pos)
        #
        # self.car_loc = temp_loc
        # self.car_last_act = temp_act
        # self.route_id = tempr_ri
        # self.alive_mask = alive_mask

        grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty)
        for road in roads:
            grid[road] = self.ROAD_CLASS


        w, h = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()
        self.pos2enc = {}
        self.enc2pos = {}
        for i in range(h):
            for j in range(w):
                if grid[i,j] != 1:
                    continue
                p = (i,j)
                # if p == (6,6):
                #     continue


                self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                v_sq = self.bool_base_grid[slice_y, slice_x][0][0].tolist()[:-1]

                self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] -= 1

                v_sq = self.list2tup(v_sq)

                #wierd bug where these get flipped, not sure why
                # if (i == 5 and j == 3):
                #     v_sq = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)

                #     v_sq = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)


                self.pos2enc[(i,j)] = v_sq
                self.enc2pos[v_sq] = (i,j)

    def obs2pos(self,obs):

        if self.enc2pos is None:
            self.init_pos2enc()

        return self.enc2pos.get(obs)

    def pos2obs(self,obs):
        if self.enc2pos is None:
            self.init_pos2enc()
        return self.pos2enc.get(tuple(obs))

    def get_is_completed(self):
        return self.has_failed, self.collision_location

    def get_alive(self,i):
        return self.alive_mask[i]

    def get_path(self,i):
        return self.chosen_path[i]

    def get_vocab_size(self):
        return self.vocab_size
