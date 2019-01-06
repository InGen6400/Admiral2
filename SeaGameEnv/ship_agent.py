import random
from typing import List, Tuple

import numpy as np

from SeaGameEnv.sea_game import Ship, NOMOVE, RIGHT, DOWN, LEFT, UP


def tank2_weighted_tank(elem):
    return (128 - abs(elem[0] - 128)) + (128 - abs(elem[1] - 128))


DIST = [[0 for i in range(256)] for j in range(256)]
for j in range(0, 256):
    for i in range(0, 256):
        DIST[j][i] = tank2_weighted_tank([j, i])
DIST = np.array(DIST)

DIST_X = [-128] * 256
for i in range(0, 128):
    DIST_X[i] = i if i != 0 else 128
    DIST_X[-i] = -i

DIST_Y = [-128] * 256
for i in range(0, 128):
    DIST_Y[i] = i if i != 0 else 128
    DIST_Y[-i] = -i

# 各探索モードの割合
MODE_PROB = np.array([5, 5, 4, 0, 3, 3])
# 和が1になるように
MODE_PROB = MODE_PROB / sum(MODE_PROB)

MODE_WEIGHTED_NEAR = 0  # スコア重み付け距離
MODE_NEAR = 1  # 距離
MODE_NEAR_BIGGEST = 2  # スコアの高いもの優先で近いもの
MODE_RANDOM = 3  # ランダム
MODE_ESCAPE_4DIR = 4  # 他の船から離れつつスコア重み付け距離
MODE_WEIGHTED_4DIR = 5  # 4方向に関して重み付けを合計して移動
MODES = np.arange(6)

QUAD_RIGHT = 0
QUAD_DOWN = 1
QUAD_LEFT = 2
QUAD_UP = 3

# 各方向マスクの定義
mask_default = np.zeros((256, 256))
y, x = np.ogrid[0:256, 0:256]
mask_r = (128 - np.abs(y - 128)) - (128 - (x - 128)) >= 0
MASK_R = mask_default.copy()
MASK_R[mask_r] = 1

mask_l = (128 - np.abs(y - 128)) - (128 + (x - 128)) > 0
MASK_L = mask_default.copy()
MASK_L[mask_l] = 1

MASK_U = np.ones((256, 256))
MASK_U[mask_r + mask_l] = 0
MASK_U[0:128, :] = 0

MASK_D = np.ones((256, 256))
MASK_D[mask_r + mask_l] = 0
MASK_D[128:256, :] = 0

QUAD_MASK = [MASK_R, MASK_D, MASK_L, MASK_U]


class ShipAgent(Ship):
    def __init__(self, name):
        self.capture = 0
        self.next_move = [NOMOVE] * 2
        self.mode = np.random.choice(MODES, p=MODE_PROB)
        self.add_name = 'm' + str(self.mode)
        super().__init__(name)

    def reset(self):
        super().reset()
        self.capture = 0
        self.next_move[0] = NOMOVE
        self.next_move[1] = NOMOVE
        self.mode = np.random.choice(MODES, p=MODE_PROB)
        self.add_name = 'm' + str(self.mode)

    def decide_move(self, ship_map, tank_map: np.ndarray):
        # 自分中心に回転
        # ship_map = np.roll(ship_map, 128-self.x, axis=1)
        # ship_map = np.roll(ship_map, 128-self.y, axis=0)
        # tank_map = np.roll(tank_map, 128-self.x, axis=1)
        # tank_map = np.roll(tank_map, 128-self.y, axis=0)

        # 自分の座標は0
        ship_map[self.pos[0]][self.pos[0]] = 0
        # 10%の確率でランダム移動
        if random.random() < 0.1:
            self.next_move[0] = random.randint(0, 4)
            self.next_move[1] = random.randint(0, 4)
        else:
            self.next_move[0] = NOMOVE
            self.next_move[1] = NOMOVE
            if self.mode == MODE_WEIGHTED_NEAR:
                self.next_move = self.decide_weighted_near(tank_map)
            elif self.mode == MODE_NEAR:
                self.next_move = self.decide_near(tank_map)
            elif self.mode == MODE_NEAR_BIGGEST:
                self.next_move = self.decide_biggest_near(tank_map)
            elif self.mode == MODE_RANDOM:
                self.next_move = self.decide_random()
            elif self.mode == MODE_ESCAPE_4DIR:
                self.next_move = self.decide_escape(ship_map, tank_map)
            elif self.mode == MODE_WEIGHTED_4DIR:
                self.next_move = self.decide_weighted_4dir(ship_map, tank_map)
            else:
                print('Unknown Decide mode: ' + str(self.mode))
                self.next_move = [NOMOVE, NOMOVE]

    def decide_weighted_near(self, tank_map):
        best_x = -1
        best_y = -1
        best_tank = 1000000
        my_x = self.pos[1]
        my_y = self.pos[0]
        y_index, x_index = np.where(tank_map != 0)
        for y, x in zip(y_index, x_index):
            tank = DIST[y - my_y][x - my_x] * 12 / tank_map[y][x]
            if tank < best_tank:
                best_tank = tank
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NOMOVE, NOMOVE]
        return self.target_to_dir([best_y, best_x])

    def decide_biggest_near(self, tank_map):
        best_x = -1
        best_y = -1
        best_dist = 10000
        best_tank = -100
        my_x = self.pos[1]
        my_y = self.pos[0]
        y_index, x_index = np.where(tank_map != 0)
        for y, x in zip(y_index, x_index):
            tank = tank_map[y][x]
            dist = DIST[y - my_y][x - my_x]
            if tank >= best_tank and dist < best_dist:
                best_tank = tank
                best_dist = dist
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NOMOVE, NOMOVE]
        return self.target_to_dir([best_y, best_x])

    def decide_near(self, tank_map):
        best_x = -1
        best_y = -1
        best_tank = 1000000
        y_index, x_index = np.where(tank_map != 0)
        my_x = self.pos[1]
        my_y = self.pos[0]
        for y, x in zip(y_index, x_index):
            tank = DIST[y - my_y][x - my_x]
            if tank < best_tank:
                best_tank = tank
                best_x = x
                best_y = y

        # タンクがないなら終了
        if best_x == -1:
            return [NOMOVE, NOMOVE]
        return self.target_to_dir([best_y, best_x])

    @staticmethod
    def decide_random():
        return [random.randint(0, 4), random.randint(0, 4)]

    def decide_escape(self, ship_map, tank_map):
        quad_score = np.zeros(4)
        my_x = self.pos[1]
        my_y = self.pos[0]
        y_index, x_index = np.where(ship_map != 0)
        for y, x in zip(y_index, x_index):
            dy = DIST_X[y - my_y]
            dx = DIST_X[x - my_x]
            quad = self.get_quadrant(dy, dx)
            quad_score[quad] = quad_score[quad] + 1

        quad_list = np.argsort(quad_score)
        best_x = -1
        best_y = -1
        best_tank = 1000000
        if y_index.size != 0:
            # TOP4 方向についてループ
            for quad_rank in range(0, 4):
                y_index, x_index = np.where(tank_map*QUAD_MASK[quad_list[quad_rank]] != 0)
                for y, x in zip(y_index, x_index):
                    tank = DIST[y - my_y][x - my_x] * 12 / tank_map[y][x]
                    if tank < best_tank:
                        best_tank = tank
                        best_x = x
                        best_y = y
                # 指定方向に(最低一つ)タンクが見つかったら終了
                if best_x != -1:
                    break
        else:
            # タンクがない
            return [NOMOVE, NOMOVE]
        return self.target_to_dir([best_y, best_x])

    def decide_weighted_4dir(self, ship_map, tank_map):
        dir_point = np.zeros(4)
        my_x = self.pos[1]
        my_y = self.pos[0]
        for quad in range(0, 4):
            # 敵船がいたら距離に応じてポイント
            y_index, x_index = np.where((ship_map * QUAD_MASK[quad]) != 0)
            for y, x in zip(y_index, x_index):
                dx = DIST_X[x - my_x]
                dy = DIST_X[y - my_y]
                # 遠くで0点 近くで4点マイナス(最遠点でx128ますy128ます = 256)
                dir_point[quad] = dir_point[quad] - (4 - 4 * ((dx + dy) / 256))

            # タンクがあったらプラスポイント
            y_index, x_index = np.where((tank_map * QUAD_MASK[quad]) != 0)
            for y, x in zip(y_index, x_index):
                dx = DIST_X[x - my_x]
                dy = DIST_X[y - my_y]
                # 遠くで容量x1点 近くで容量x4点プラス
                dir_point[quad] = dir_point[quad] + (4 - 4 * ((dx + dy) / 256)) * tank_map[y][x] * 5
        dir = dir_point.argmax()
        return [dir, dir]

    def target_to_dir(self, target_pos: List[float]) -> List[str]:
        ret = [NOMOVE, NOMOVE]
        dx = target_pos[1] - self.pos[1]
        dy = target_pos[0] - self.pos[0]
        # X移動のほうが遠い
        if abs(dx) > abs(dy):
            if dx < 0:
                ret[0] = LEFT
            else:
                ret[0] = RIGHT
            dx = dx - 10
        else:
            if dy < 0:
                ret[0] = DOWN
            else:
                ret[0] = UP
            dy = dy - 10

        # 二回目の移動
        if abs(dx) < 10 and abs(dy) < 10:
            ret[1] = NOMOVE
            if abs(dx) > abs(dy):
                if dx < 0:
                    ret[1] = LEFT
                else:
                    ret[1] = RIGHT
        return ret

    @staticmethod
    def get_quadrant(y, x):
        if MASK_R[y+128][x+128] == 1:
            return QUAD_RIGHT
        elif MASK_D[y+128][x+128] == 1:
            return QUAD_DOWN
        elif MASK_L[y+128][x+128] == 1:
            return QUAD_LEFT
        elif MASK_U[y+128][x+128] == 1:
            return QUAD_UP
        else:
            print('Out of QUad')
