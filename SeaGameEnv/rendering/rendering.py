import sys
import pygame
from asyncio import sleep
from numpy.core.multiarray import ndarray
from typing import List, Any

from pygame.locals import *

import numpy as np

class Render(object):
    def __init__(self, width, height, scale, title='', ship_size=14, tank_size=10):
        pygame.init()
        self.size = (width*scale, height*scale)
        self.window = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.game_surface = pygame.Surface((256*scale, 256*scale))
        self.ship_font = pygame.font.Font(None, 16*scale)
        self.tank_font = pygame.font.Font(None, 12*scale)
        self.scale = scale
        self.ship_size = ship_size*scale
        self.tank_size = tank_size*scale

    def update(self):
        # 上下反転
        self.window.blit(pygame.transform.flip(self.game_surface, False, True), (0, 0))
        pygame.display.update()
        self.game_surface.fill((0, 0, 255))
        for event in pygame.event.get():
            if event.type == QUIT:          # 閉じるボタンが押されたとき
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:       # キーを押したとき
                if event.key == K_ESCAPE:   # Escキーが押されたとき
                    pygame.quit()
                    sys.exit()

    def draw_ship(self, ship_list: List[Any]):
        is_admiral = True
        for ship in ship_list:
            if is_admiral:
                ellipse_color = (128, 200, 0)
                is_admiral = False
            else:
                ellipse_color = (255, 0, 0)
            pygame.draw.ellipse(self.game_surface, ellipse_color, ((ship.pos[1]*self.scale-self.ship_size/2),
                                                                 (ship.pos[0]*self.scale-self.ship_size/2),
                                                                 self.ship_size, self.ship_size))
            text_view = self.ship_font.render(ship.name, True, (255, 255, 255))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (ship.pos[1]*self.scale+self.ship_size/2, ship.pos[0]*self.scale))
            text_view = self.ship_font.render(str(ship.point), True, (0, 255, 0))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (ship.pos[1]*self.scale+self.ship_size/2, ship.pos[0]*self.scale-self.ship_size))

    def draw_tank(self, tank_list: List[Any]):
        for tank in tank_list:
            pygame.draw.ellipse(self.game_surface, (255, 255, 0), ((tank.pos[1]*self.scale-self.tank_size/2),
                                                                   (tank.pos[0]*self.scale-self.tank_size/2),
                                                                   self.tank_size, self.tank_size))
            text_view = self.tank_font.render(str(tank.point), True, (0, 0, 0))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (tank.pos[1]*self.scale - text_view.get_rect().width/2,
                                    tank.pos[0]*self.scale - text_view.get_rect().height/2))

    def draw_ship_map(self, ship_map: ndarray):
        y_index, x_index = np.where(ship_map != 0)
        for y, x in zip(y_index, x_index):
            pygame.draw.ellipse(self.game_surface, (255, 0, 0), ((x*self.scale-self.ship_size/2),
                                                                 (y*self.scale-self.ship_size/2),
                                                                 self.ship_size, self.ship_size))
            text_view = self.ship_font.render(str(ship_map[y][x]), True, (0, 255, 0))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (x*self.scale+self.ship_size/2, y*self.scale-self.ship_size))

    def draw_tank_map(self, tank_map: ndarray):
        y_index, x_index = np.where(tank_map != 0)
        for y, x in zip(y_index, x_index):
            pygame.draw.ellipse(self.game_surface, (255, 255, 0), ((x*self.scale-self.tank_size/2),
                                                                   (y*self.scale-self.tank_size/2),
                                                                   self.tank_size, self.tank_size))
            text_view = self.tank_font.render(str(tank_map[y][x]), True, (0, 0, 0))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (x*self.scale - text_view.get_rect().width/2,
                                    y*self.scale - text_view.get_rect().height/2))
