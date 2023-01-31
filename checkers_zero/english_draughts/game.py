import numpy as np
from checkers_zero.environment import Environment
from .state import EnglishDraughtsState
from typing import Any
class EnglishDraughtsEnv(Environment):
    def __init__(self,all_kings_mode=False) -> None:
        super().__init__()
        self._state = self._initialize_new_state(all_kings_mode)
        self._all_kings = all_kings_mode
        self.main_window = None # type: ignore

    @property
    def n_actions(self) -> int:
        return EnglishDraughtsState.N_ACTIONS

    @property
    def observation_space(self) -> tuple:
        return self._state.observation_space

    def reset(self) -> EnglishDraughtsState:
        self._state = self._initialize_new_state(self._all_kings)
        return self._state

    def step(self, action: int) -> tuple[EnglishDraughtsState, bool]:
        self._state = self._state.step(action)
        terminal = self._state.is_terminal()
        return self._state, terminal

    def render(self) -> None:
        import pygame
        from checkers_zero.constants import WHITE_COLOR,RED_COLOR,BLUE_COLOR,BLACK_COLOR,GRAY_COLOR
        if self.main_window is None:
            pygame.init()
            pygame.display.init()
            self.main_win_dims  = self.main_width , self.main_height = 500,500
            self.main_window = pygame.display.set_mode(self.main_win_dims)
            self.game_surface = self.main_window.subsurface((5,5,self.main_width-5,self.main_height-5))
            self.cell_size = self.game_surface.get_width() // 8
            self._crown = pygame.Surface((self.cell_size * 3 //4//2,self.cell_size * 3 //4//2),pygame.SRCALPHA,32)
            sz = 4
            pygame.draw.rect(self._crown,WHITE_COLOR,(self._crown.get_width()//2-sz//2,0,sz,self._crown.get_height()))
            pygame.draw.rect(self._crown,WHITE_COLOR,(0,self._crown.get_height()//2-sz//2,self._crown.get_width(),sz))
            self._clock = pygame.time.Clock()
        self._draw_board()
        pygame.event.pump()
        self._clock.tick(50)

        pygame.display.update()

    def player_turn(self) -> int:
        return self._state.player_turn()

    
    def _draw_board(self):
        import pygame
        from checkers_zero.constants import WHITE_COLOR,RED_COLOR,BLACK_COLOR,BLUE_COLOR,GRAY_COLOR
        pygame.draw.rect(self.game_surface,WHITE_COLOR,(0,0,self.game_surface.get_width(),self.game_surface.get_height()))
        for row in range(8):
            top_offset = row*self.cell_size
            for col in range(8):
                left_offset = col * self.cell_size
                if (row+col) % 2 == 1:
                    pygame.draw.rect(
                        self.game_surface, RED_COLOR, (top_offset, left_offset, self.cell_size, self.cell_size))
        obs = self._state.to_obs()
        players_colors = [BLUE_COLOR,BLACK_COLOR]
        player = self._state.player_turn()
        for row in range(8):
            top_offset = row*self.cell_size
            for col in range(8):
                left_offset = col * self.cell_size
                if player == 1:
                    relative_row = 8-row-1
                    relative_col = 8-col-1
                else:
                    relative_row = row
                    relative_col = col
                if obs[0, relative_row, relative_col] == 1:
                    player_color = players_colors[player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                elif obs[1, relative_row, relative_col] == 1:
                    # TODO draw king
                    center = (left_offset+self.cell_size//2, top_offset+self.cell_size//2)                    
                    player_color = players_colors[player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                    self.game_surface.blit(self._crown,(center[0]-self._crown.get_width()//2,center[1]-self._crown.get_height()//2))
                elif obs[2, relative_row, relative_col] == 1:
                    player_color = players_colors[1-player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                elif obs[3, relative_row, relative_col] == 1:
                    # TODO draw king
                    center = (left_offset+self.cell_size//2, top_offset+self.cell_size//2)  
                    player_color = players_colors[1-player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                    self.game_surface.blit(self._crown,(center[0]-self._crown.get_width()//2,center[1]-self._crown.get_height()//2))

    def _initialize_new_state(self,all_kings:bool) -> EnglishDraughtsState:
        observation = np.array([
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]
        ], dtype=np.int32)
        if all_kings:
            observation = observation*2
        obs_ar = np.zeros((4, EnglishDraughtsState.ROWS,
                          EnglishDraughtsState.COLS), dtype=np.int32)
        obs_ar[0, :, :] = observation == 1
        obs_ar[1, :, :] = observation == 2
        obs_ar[2, :, :] = observation == -1
        obs_ar[3, :, :] = observation == -2
        state = EnglishDraughtsState(obs_ar, 0, dict(), None, None, 0)
        return state