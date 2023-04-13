import copy
from ctypes.wintypes import CHAR
import logging
from pickle import LONG4
import gym
import os
import torch as T
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding
from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_score_board
from ..utils.observation_space import MultiAgentObservationSpace


"""
Minigrid Environment for Automata Project with multi-agent systems.

LIMITATIONS OF AGENT/S:
Each agent's observation includes its:
    - Agent ID 
    - Position within the grid
    - Number of steps since beginning
    - Full Observability of the environment

------------------------------------------------------------------------------------------------------------------------------

ACTION SPACE:
------------------------------------------------------------------------------------------------------------------------------

{move forward, move left, move right, break, craft}

Shortened to:
{F, R, L, B, C}

------------------------------------------------------------------------------------------------------------------------------

Arguments:
    grid_shape: size of the grid
    n_agents: 
    n_rocks:
    n_fires:
    n_trees:

    agent_view: size of the agent view range in each direction
    full_observable: flag whether agents should receive observation for all other agents

Attributes:
    _agent_dones: list with indicater whether the agent is done or not.
    _base_img: base image with grid
    _viewer: viewer for the rendered image
"""


class MinigridRock(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']} #must be human so it is readable data by humans

    def __init__(self, grid_shape=(10, 10), n_agents=2, n_rocks=1, n_fires = 5, n_trees = 1, agents_id = 'ab', goal = "T", full_observable=True, max_steps=300, agent_view_mask=(10,10)):
        
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert len(agent_view_mask) == 2, 'expected a tuple of size 2 for agent view mask,' \
                                          ' but found {}'.format(agent_view_mask)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        assert 0 < agent_view_mask[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        assert 0 < agent_view_mask[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])

        ############# PARAMS #################
        self._base_grid_shape = grid_shape
        self._agent_grid_shape = agent_view_mask

        self.n_agents = n_agents
        self.n_rocks = n_rocks
        self.n_fires = n_fires
        self.n_trees = n_trees

        self._max_steps = max_steps
        self._step_count = None
        self._agent_view_mask = agent_view_mask
        self.agent_reward = 0
        self.agent_action = None

        self.agents_id = agents_id
        self.goal = goal
        self.observation_cntr = 0

        self.AGENT_COLOR = ImageColor.getcolor("blue", mode='RGB')
        self.OBSERVATION_VISUAL_COLOR = ImageColor.getcolor("gray", mode='RGB')
        self.ROCK_COLOR = ImageColor.getcolor("gray", mode='RGB')
        self.FIRE_COLOR = ImageColor.getcolor("red", mode='RGB')
        self.TREE_COLOR = ImageColor.getcolor("green", mode='RGB')
        self.WALL_COLOR = ImageColor.getcolor("black", mode='RGB')
        self.DEBRIS_COLOR = ImageColor.getcolor("saddlebrown", mode='RGB')

        # self.CRAFTING_COLOR = ImageColor.getcolor("yellow", mode='RGB')

        #Inital position Initilization for each object
        self._init_agent_pos = {_: None for _ in range(self.n_agents)}
        self._init_rock_pos = {_: None for _ in range(self.n_rocks)}
        self._init_fire_pos = {_: None for _ in range(self.n_fires)}
        self._init_tree_pos = {_: None for _ in range(self.n_trees)}

        self._base_grid = self.__create_grid() 
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])

        self.viewer = None
        self.full_observable = full_observable

        mask_size = np.prod(self._agent_view_mask)

        self._obs_high = np.array([5]*mask_size + 2*[5.0], dtype=np.float32)
        self._obs_low  = np.array([0]*mask_size + 2*[0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()
    """
        Function: get_action_meanings()
        Inputs  : agent_i 
        Outputs : None
        Purpose : Reports back what move agent/user chose
    """
    # def get_action_meanings(self, agent_i=None):
    #     if agent_i is not None:
    #         assert agent_i <= self.n_agents
    #         return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
    #     else:
    #         return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def sample_action_space(self): # only for eps-greedy action  
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self): # draws empty grid
        self._base_img = draw_grid(self._base_grid_shape[0], self._base_grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._base_grid_shape[1])] for row in range(self._base_grid_shape[0])]
        return _grid

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.rock_pos = {}
        self.fire_pos = {}
        self.tree_pos = {}  
        self.wall_pos = {}
        self.debris_pos = {}

        self.__init_map()
        
        self.inventory = {'tree': 0, 'rock':0}
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    def __init_map(self):
        self._full_obs = self.__create_grid()
        
        pos = [[int(self._base_grid_shape[0]/2),int(self._base_grid_shape[0]/2)-3],[int(self._base_grid_shape[0]/2),int(self._base_grid_shape[0]/2)-2],[int(self._base_grid_shape[0]/2),int(self._base_grid_shape[0]/2)+2],[int(self._base_grid_shape[0]/2),int(self._base_grid_shape[0]/2)+3]]
        self.n_debris = len(pos)
        debris_counter = 0
        for p in pos:
            if self._is_cell_vacant(p):
                self.debris_pos[debris_counter] = p
            self.__update_debris_view(debris_counter)                    
            debris_counter += 1

        wall_counter = 0
        for col in range(self._base_grid_shape[0]):
            pos = [int(self._base_grid_shape[0]/2),col]
            if self._is_cell_vacant(pos):
                self.wall_pos[wall_counter] = pos
                self.__update_wall_view(wall_counter)                    
                wall_counter+= 1

        self.n_walls = wall_counter

        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._base_grid_shape[0]/2 - 1), 
                       self.np_random.randint(0, self._base_grid_shape[1]/2 - 1)]
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    self._init_agent_pos = pos
                    break
            self.__update_agent_view(agent_i)
        
     
        for rock_i in range(self.n_rocks):
            while True:
                pos = [self.np_random.randint(0, self._base_grid_shape[0]/2 - 1),
                       self.np_random.randint(0, self._base_grid_shape[1]/2 - 1)]
                if self._is_cell_vacant(pos):
                    self.rock_pos[rock_i] = pos
                    self._init_rock_pos = pos
                    break
            self.__update_rock_view(rock_i)

        for fire_i in range(self.n_fires):
            while True:
                pos = [self.np_random.randint(0, self._base_grid_shape[0] - 1),
                       self.np_random.randint(0, self._base_grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and not pos[0] == int(self._base_grid_shape[0]/2)-1 and not pos[1] == int(self._base_grid_shape[0]/2)-3 \
                      and not pos[1] == int(self._base_grid_shape[0]/2)-2 and not pos[1] == int(self._base_grid_shape[0]/2)+3 and not pos[1] == int(self._base_grid_shape[0]/2)+2:
                    self.fire_pos[fire_i] = pos
                    self._init_fire_pos = pos
                    break
            self.__update_fire_view(fire_i)

        for tree_i in range(self.n_trees):
            while True:
                pos = [self.np_random.randint(0, self._base_grid_shape[0]/2 - 1),
                       self.np_random.randint(0, self._base_grid_shape[1]/2 - 1)]
                if self._is_cell_vacant(pos):
                    self.tree_pos[tree_i] = pos
                    self._init_tree_pos = pos
                    break
            self.__update_tree_view(tree_i)



        self.__draw_base_img()
    """
        Function : get_agent_obs()
        Inputs   : None
        Outputs  : full observation of the grid world
        Purpose  : 
    """
    # def get_agent_obs(self):
    #     _obs = []
    #     for agent_i in range(self.n_agents):
    #         pos = self.agent_pos[agent_i]
    #         # print("Agent Pos: {}".format(pos))

    #         _agent_i_obs = [pos[0] / (self._agent_grid_shape[0] - 1), pos[1] / (self._agent_grid_shape[1] - 1)]  #coordinate of agent

    #         # check if rock is in the view area and give it future (time+1) rock coordinates
    #         _rock_pos = np.zeros(self._agent_view_mask)  
    #         _fire_pos = np.zeros(self._agent_view_mask) # rock location in neighbor
    #         for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._agent_grid_shape[0] - 2)):
    #             for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._agent_grid_shape[1] - 2)):
    #                 if PRE_IDS['rock'] in self._full_obs[row][col]:
    #                     _rock_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  

    #         _agent_i_obs += _rock_pos.flatten().tolist()  # adding rock pos in observable area
    #         _agent_i_obs += _fire_pos.flatten().tolist()
    #         _agent_i_obs += [self._step_count / self._max_steps]  # adding the time

    #         _obs.append(_agent_i_obs)

    #     if self.full_observable:
    #         _obs = np.array(_obs).flatten().tolist() # flatten to np array 
    #         _obs = [_obs for _ in range(self.n_agents)]
    #     return _obs

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            for row in range(len(self._full_obs[0])):
                for col in range(len(self._full_obs[1])):
                    _obs.append(self._full_obs[row][col])
            _obs.append(self.inventory['tree'])
            _obs.append(self.inventory['rock'])
        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist() # flatten to np array 
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def is_valid(self, pos):
        return (0 <= pos[0] < self._agent_grid_shape[0]) and (0 <= pos[1] < self._agent_grid_shape[1])

    def _is_cell_vacant(self, pos):
        if self.is_valid(pos):
            return (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']) 

    def _is_next_pos_fire(self,pos):
        if self.is_valid(pos):
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['fire']:
                return True
        return False

    def _is_next_pos_debris(self,pos):
        if self.is_valid(pos):
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['debris']:
                return True
        return False

    def _is_next_pos_wall(self,pos):
        if self.is_valid(pos):
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['wall']:
                return True
        return False

    def _is_agent_next_to_tree(self,pos):
        next_to_tree = False
        which_tree = []
        for tree_i in self.tree_pos:
            # print(self.tree_pos[tree_i])
            if [pos[0]-1,pos[1]] == self.tree_pos[tree_i] or [pos[0]+1,pos[1]] == self.tree_pos[tree_i] or [pos[0],pos[1]+1] == self.tree_pos[tree_i] or [pos[0],pos[1]-1] == self.tree_pos[tree_i]:            
                next_to_tree = True
                which_tree.append(self.tree_pos[tree_i])
        return next_to_tree, which_tree

    def _is_agent_next_to_rock(self,pos):
        next_to_rock = False
        which_rock = []
        for rock_i in self.rock_pos:
            # print(self.rock_pos[rock_i])
            if [pos[0]-1,pos[1]] == self.rock_pos[rock_i] or [pos[0]+1,pos[1]] == self.rock_pos[rock_i] or [pos[0],pos[1]+1] == self.rock_pos[rock_i] or [pos[0],pos[1]-1] == self.rock_pos[rock_i]:            
                next_to_rock = True
                which_rock.append(self.rock_pos[rock_i])
        return next_to_rock, which_rock

    def __update_agent_pos(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None

        moves = {
            0: [curr_pos[0]-1, curr_pos[1]],   #Move up
            1: [curr_pos[0], curr_pos[1] - 1], # move left
            2: [curr_pos[0], curr_pos[1] + 1], # move right
            3: [curr_pos[0] + 1, curr_pos[1]], # move down
            4: [curr_pos[0], curr_pos[1]]      #break
        }

        next_pos = moves[move]

        if self._is_next_pos_fire(next_pos):
            # print("hit fire")
            return True

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)
            # self.update_agent_color([move])

        return False


    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + agent_i

    def __update_rock_view(self, rock_i):
        self._full_obs[self.rock_pos[rock_i][0]][self.rock_pos[rock_i][1]] = PRE_IDS['rock']

    def __update_fire_view(self, fire_i):
        self._full_obs[self.fire_pos[fire_i][0]][self.fire_pos[fire_i][1]] = PRE_IDS['fire']

    def __update_tree_view(self, tree_i):
        self._full_obs[self.tree_pos[tree_i][0]][self.tree_pos[tree_i][1]] = PRE_IDS['tree']

    def __update_debris_view(self, debris_i):
        self._full_obs[self.debris_pos[debris_i][0]][self.debris_pos[debris_i][1]] = PRE_IDS['debris']

    def __update_wall_view(self, wall_i):
        self._full_obs[self.wall_pos[wall_i][0]][self.wall_pos[wall_i][1]] = PRE_IDS['wall']


    def __delete_item(self, pos):
        self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']


    """
        Function : step
        Purpose  : This function returns the rewards, state, and terminal conditions for each action decision
    """   
    def step(self, agents_action):
        rewards = [0 for _ in range(self.n_agents)]
        if (self._step_count >= self._max_steps):
            for i in range(self.n_agents):
                self._agent_dones[i] = True
            return self.get_agent_obs(), rewards, self._agent_dones, self.observation_cntr

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                is_hitting_fire = self.__update_agent_pos(agent_i, action)
                if is_hitting_fire:
                    for i in range(self.n_agents):
                        self._agent_dones[i] = True
                    return self.get_agent_obs(), rewards, self._agent_dones, self.observation_cntr                    
        if all(action == 5 for action in agents_action):
            item_breakable, pos = self._check_if_item_breakable()
            if item_breakable:
                self.inventory['rock'] += 1 
                self.__delete_item(pos)
                rewards = [1 for _ in range(self.n_agents)]                   
                for i in range(self.n_agents):
                    self._agent_dones[i] = True
                return self.get_agent_obs(), rewards, self._agent_dones, self.observation_cntr

        self._step_count += 1
         
        return self.get_agent_obs(), rewards, self._agent_dones, self.observation_cntr

    def _check_if_item_breakable(self):
            next_to_rock_flag = False
            which_rock_old = []
            which_rock_new = []
            which_rock = None
            for agent_i in range(self.n_agents):
                next_rock, which_rock_new =  self._is_agent_next_to_rock(self.agent_pos[agent_i])
                if next_rock and len(which_rock_new)>0:
                    if len(which_rock_old) == 0:
                        which_rock_old = which_rock_new
                    else:
                        for which_rock_i in which_rock_old:
                            if which_rock_i in which_rock_new:
                                next_to_rock_flag = True
                                which_rock = which_rock_i
                                break
                else:
                    next_to_rock_flag = False
                    return False, None
            return next_to_rock_flag, which_rock

    def render(self, mode='human'): #renders the entire grid world as one frame
        
        img = copy.copy(self._base_img)

        #agent visual
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=self.AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
        #rock visual
        for rock_i in range(self.n_rocks):
            draw_circle(img, self.rock_pos[rock_i], cell_size=CELL_SIZE, fill=self.ROCK_COLOR)   
            write_cell_text(img, text=str(rock_i + 1), pos=self.rock_pos[rock_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
            
        for fire_i in range(self.n_fires):
            draw_circle(img, self.fire_pos[fire_i], cell_size=CELL_SIZE, fill=self.FIRE_COLOR)   
            write_cell_text(img, text=str(fire_i + 1), pos=self.fire_pos[fire_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
            
        for tree_i in range(self.n_trees):
            draw_circle(img, self.tree_pos[tree_i], cell_size=CELL_SIZE, fill=self.TREE_COLOR)   
            write_cell_text(img, text=str(tree_i + 1), pos=self.tree_pos[tree_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for debris_i in range(self.n_debris):
            draw_circle(img, self.debris_pos[debris_i], cell_size=CELL_SIZE, fill=self.DEBRIS_COLOR)   
            write_cell_text(img, text=str(debris_i + 1), pos=self.debris_pos[debris_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for wall_i in range(self.n_walls):
            draw_circle(img, self.wall_pos[wall_i], cell_size=CELL_SIZE, fill=self.WALL_COLOR)   
            write_cell_text(img, text=str(wall_i + 1), pos=self.wall_pos[wall_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        # img = draw_score_board(img,[self.agent_reward,self.agent_action])
        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

CELL_SIZE = 35
WALL_COLOR = 'white'

#sep. actions for observing and slewing 
# PRE_IDS = {
#     'agent': 'A',
#     'rock': 'R',
#     'wall': 'W',
#     'empty': '0',
#     'fire' : 'F',
#     'tree' : 'T'
# }

# PRE_IDS = {
#     'agent': '4',
#     'rock': '1',
#     'wall': 'W',
#     'empty': '0',
#     'fire' : '3',
#     'tree' : '2'
# }

PRE_IDS = {
    'agent': 6,
    'rock': 1,
    'wall': 4,
    'empty': 0,
    'fire' : 3,
    'tree' : 2,
    'debris':5

}