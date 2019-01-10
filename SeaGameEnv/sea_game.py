import random
from time import sleep
from typing import List

import numpy as np
import skimage.measure
import gym
import skimage
from gym import spaces, logger
from numba import jit
from numpy.core.multiarray import ndarray


DIR_2_VECTOR: ndarray = np.array([
    [0, 10],
    [-10, 0],
    [0, -10],
    [10, 0],
    [0, 0]
])

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
NOMOVE = 4

PENARTY = 0.1
NO_PENALTY_MAX = 2


class Tank(object):
    def __init__(self, point, y, x):
        self.point = point
        self.pos = (y, x)


class Ship(object):
    pos: ndarray

    def __init__(self, name):
        self.name = name
        self.point = 0
        self.capture = 0
        self.pos = np.array([random.randrange(0, 256), random.randrange(0, 256)])
        #self.pos = np.array([128, 128])

    def reset(self):
        self.point = 0
        self.capture = 0
        self.pos = np.array([random.randrange(0, 256), random.randrange(0, 256)])
        #self.pos = np.array([128, 128])

    def move(self, moves: List[int]):
        self.pos = self.pos + DIR_2_VECTOR[moves[0]]
        self.pos = self.pos + DIR_2_VECTOR[moves[1]]
        # 0~256に収める
        if self.pos[0] < 0:
            self.pos[0] = self.pos[0]+256
        if self.pos[1] < 0:
            self.pos[1] = self.pos[1]+256
        self.pos[0] = self.pos[0] % 256
        self.pos[1] = self.pos[1] % 256


from SeaGameEnv.rendering.rendering import Render
from SeaGameEnv.ship_agent import ShipAgent


class SeaGameEnv(gym.core.Env):
    tank_map: ndarray
    ship_map: ndarray
    npc_list: List[ShipAgent]
    ship_list: List[Ship]
    tank_list: List[Tank]

    def __init__(self, nb_npc=5, max_step=None, npc_name='*', player_name='Admiral', ship_pool=8, tank_pool=8):
        self.map = np.zeros((256, 512))
        self.ship_map = self.map[:, :256]
        self.tank_map = self.map[:, 256:]
        self.tank_list = []
        self.tank_plan = []
        self.tank_all = 0
        self.nb_step = 0
        self.no_penalty_step = 0
        self.is_test = False
        self.seed_value = None
        self.ship_pool = ship_pool
        self.tank_pool = tank_pool
        if max_step:
            self.max_step = max_step
        else:
            self.max_step = random.randrange(1, 5)*60*2
        self.nb_npc = nb_npc
        self.screen = Render(256, 256, 2)
        self.ship_list = [Ship(player_name)] + [ShipAgent(npc_name + str(i + 1)) for i in range(nb_npc)]

        self.action_space = gym.spaces.Discrete(len(ACTION_MEANS))
        #self.observation_space = gym.spaces.Box(low=0, high=1,
        #                                        shape=(256//self.ship_pool + 256//self.tank_pool,
        #                                               256//self.ship_pool + 256//self.tank_pool, 1), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=((256//self.ship_pool)**2 + (256//self.tank_pool)**2,), dtype=np.int64)
        self.obs = self.observe()

    def step(self, action: int):
        # AI移動
        moves = ACTION_MEANS[action]
        self.ship_list[0].move(moves)

        # NPC移動
        ship: ShipAgent
        for ship in self.ship_list[1:]:
            ship.decide_move(self.ship_list, self.tank_list)
            ship.move(ship.next_move)

        self.collide()

        done = False
        self.nb_step = self.nb_step + 1
        if self.nb_step >= self.max_step:
            done = True
        else:
            # 2ステップに一回タンクを生成
            if self.nb_step % 2 == 0:
            #if len(self.tank_list) == 0:
                self.tank_list.append(Tank(self.tank_plan.pop(0), random.randrange(0, 256), random.randrange(0, 256)))

        self.mapping()
        self.no_penalty_step = self.no_penalty_step - 1
        # 獲得したなら
        if self.ship_list[0].capture != 0:
            # ペナルティを一定期間免除
            if self.no_penalty_step > 0:
                # すでに免除期間なら半分延長
                self.no_penalty_step = self.no_penalty_step + NO_PENALTY_MAX/2
            else:
                # ペナルティ免除期間設定
                self.no_penalty_step = NO_PENALTY_MAX
        if self.no_penalty_step > 0:
            # ペナルティ免除中
            reward = self.ship_list[0].capture / self.tank_all * 1000
        else:
            # ペナルティを課す
            reward = self.ship_list[0].capture / self.tank_all * 1000 - PENARTY
        self.ship_list[0].capture = 0
        self.obs = self.observe()
        return self.obs, reward, done, {}

    def collide(self):
        for ship in self.ship_list:
            for tank in self.tank_list:
                dx = tank.pos[1] - ship.pos[1]
                dy = tank.pos[0] - ship.pos[0]
                if dx > 128:
                    dx = dx - 256
                elif dx < -128:
                    dx = dx + 256
                if dy > 128:
                    dy = dy - 256
                elif dy < -128:
                    dy = dy + 256
                if dx*dx+dy*dy < 100:
                    ship.point = ship.point + tank.point
                    ship.capture = tank.point
                    self.tank_list.remove(tank)

    def mapping(self):
        self.ship_map.fill(0)
        for ship in self.ship_list:
            self.ship_map[(ship.pos[0], ship.pos[1])] = self.ship_map[(ship.pos[0], ship.pos[1])] + 1

        self.tank_map.fill(0)
        for tank in self.tank_list:
            self.tank_map[tank.pos] = self.tank_map[tank.pos] + tank.point

    def get_map(self):
        x = self.ship_list[0].pos[1]
        y = self.ship_list[0].pos[0]
        ship_map = np.roll(self.ship_map, 128-x, axis=1)
        ship_map = np.roll(ship_map, 128-y, axis=0)
        tank_map = np.roll(self.tank_map, 128-x, axis=1)
        tank_map = np.roll(tank_map, 128-y, axis=0)
        return ship_map, tank_map

    def observe(self):
        ship_map, tank_map = self.get_map()

        XK = 256//self.ship_pool
        ship_map = ship_map[:XK*self.ship_pool, :XK*self.ship_pool]\
            .reshape(XK, self.ship_pool, XK, self.ship_pool)\
            .sum(axis=(1, 3))
        XK = 256//self.tank_pool
        tank_map = tank_map[:XK*self.tank_pool, :XK*self.tank_pool]\
            .reshape(XK, self.tank_pool, XK, self.tank_pool)\
            .sum(axis=(1, 3))

        #self.screen.draw_ship_map(ship_map)
        #self.screen.draw_tank_map(tank_map)
        # 一列に並べるflatten
        return np.hstack((ship_map.reshape([(256//self.ship_pool)**2]), tank_map.reshape([(256//self.tank_pool)**2])))

    def seed(self, seed=None):
        random.seed(seed)
        self.seed_value = seed

    def reset(self):

        with open('match.log', 'a') as f:
            ships = sorted(self.ship_list, key=lambda x: x.point)
            print('\n', file=f)
            i = 0
            for ship in ships:
                if type(ship) == ShipAgent:
                    ship: ShipAgent
                    print(str(i+1) + ': ' + ship.name + ship.add_name + '/ ' + str(ship.point), file=f)
                else:
                    print(str(i+1) + ': ' + ship.name + '/ ' + str(ship.point), file=f)
                i = i+1

        self.nb_step = 0
        self.map.fill(0)
        self.tank_list.clear()
        self.tank_plan.clear()
        for _ in range(self.max_step//2):
            self.tank_plan.append(random.randint(1, 4))
        self.tank_all = sum(self.tank_plan)
        for ship in self.ship_list:
            ship.reset()
        self.mapping()
        return self.observe()

    def render(self, mode='human'):
        self.screen.draw_ship(self.ship_list)
        self.screen.draw_tank(self.tank_list)
        if self.is_test:
            sleep(0.2)
        self.screen.update()


ACTION_MEANS = [
    [RIGHT, RIGHT],
    [RIGHT, DOWN],
    [RIGHT, LEFT],
    [RIGHT, UP],
    [DOWN, RIGHT],
    [DOWN, DOWN],
    [DOWN, LEFT],
    [DOWN, UP],
    [LEFT, RIGHT],
    [LEFT, DOWN],
    [LEFT, LEFT],
    [LEFT, UP],
    [UP, RIGHT],
    [UP, DOWN],
    [UP, LEFT],
    [UP, UP],
]
