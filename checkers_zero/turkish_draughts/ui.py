import pygame
import numpy as np
from .state import TurkishDraughtsState
from checkers_zero.constants import BLUE_COLOR, WHITE_COLOR, BLACK_COLOR, RED_COLOR, GRAY_COLOR
SCREEN_WIDTH, SCREEN_HEIGHT = 500, 500

class TurkishDraughtsUI():
    def __init__(self,game_surface:pygame.surface.Surface,ai_path:str) -> None:
        pygame.font.init()
        