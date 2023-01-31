import pygame
import numpy as np
from .state import EnglishDraughtsState
from .game import EnglishDraughtsEnv
from checkers_zero.constants import BLUE_COLOR, WHITE_COLOR, BLACK_COLOR, RED_COLOR, GRAY_COLOR
SCREEN_WIDTH, SCREEN_HEIGHT = 500, 500

class EnglishDraughtsUI():
    def __init__(self, game_surface: pygame.surface.Surface,ai_path:str) -> None:
        pygame.font.init()
        self.env = EnglishDraughtsEnv()
        self.game_surface = game_surface
        self.cell_size = game_surface.get_width() // 8
        self.env.reset()
        self.terminal = False
        self.reward: np.ndarray | None = None

        self.freeze_until = 0
        self.selected = None
        self.player = self.env.player_turn()
        self._crown = pygame.Surface((self.cell_size * 3 //4//2,self.cell_size * 3 //4//2),pygame.SRCALPHA,32)
        sz = 4
        pygame.draw.rect(self._crown,WHITE_COLOR,(self._crown.get_width()//2-sz//2,0,sz,self._crown.get_height()))
        pygame.draw.rect(self._crown,WHITE_COLOR,(0,self._crown.get_height()//2-sz//2,self._crown.get_width(),sz))
        
        from checkers_zero.networks import SharedResNetwork
        from checkers_zero.players import AMCTSPlayer
        network = SharedResNetwork(self.env.observation_space,self.env.n_actions)
        network.load_model(ai_path)
        self.ai = AMCTSPlayer(self.env.n_actions,network,1,1000,1,1)
        self.is_evaluating = False

    def render(self):
        if self.terminal:
            if pygame.time.get_ticks() > self.freeze_until:
                return False
            else:
                return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                player = self.env.player_turn()
                if player != 1:
                    continue
                pos = pygame.mouse.get_pos()
                print(pos)
                coordinates = self._get_row_col_from_pos(pos)
                if coordinates is None:
                    continue
                if self.selected is not None:
                    if self.selected != coordinates:
                        action = self._encode_action(
                            self.selected, coordinates)
                        if action is None:
                            continue
                        try:
                            state, self.terminal = self.env.step(action)
                            self.player = self.env.player_turn()
                            
                            if self.terminal:
                                self.reward = state.game_result()
                                self.freeze_until = pygame.time.get_ticks()+5000
                        except ValueError as v:
                            continue
                        finally:
                            self.selected = None
                    else:
                        self.selected = None
                else:
                    self.selected = coordinates
        self._draw_game()
        if self.terminal:
            self.freeze_until = pygame.time.get_ticks()+5000
        elif self.player == 0 and not self.is_evaluating:
            try:
                
                action = self.ai.choose_action(self.env._state)
                state, self.terminal = self.env.step(action)
                self.player = self.env.player_turn()
            except ValueError as v:
                ...
        return True

    def _draw_game(self):
        self._draw_board()
        if self.terminal:
            self._draw_winner_name()

    def _draw_winner_name(self):
        f = pygame.font.SysFont("comicsans", self.cell_size//5, bold=True)
        player_name = " 1 "
        if self.reward is None:
            raise ValueError("reward is None")
        reward = self.reward
        if self.player == 1:
            reward = self.reward[::-1]

        if reward[0] == 1:
            player_name = " 1 "
        elif reward[2] == 1:
            player_name = " 2 "
        text = f.render(f"Player {player_name} has won.", True, RED_COLOR)
        if self.reward[1] == 1:
            text = f.render(f"Draw", True, RED_COLOR)

        self.game_surface.blit(
            text, (self.game_surface.get_width()//2 - text.get_width()//2, 0))

    def _get_row_col_from_pos(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        x, y = pos
        s_x, s_y = self.game_surface.get_offset()
        x, y = x-s_x, y-s_y
        
        # print(f'{(x,y)} : {(s_x,s_y)}')

        if x > self.game_surface.get_width() or y > self.game_surface.get_height():
            return None

        row = y // self.cell_size
        col = x // self.cell_size
        padding = self.cell_size//8
        min_x = col * self.cell_size + padding
        max_x = (col+1) * self.cell_size - padding
        min_y = row * self.cell_size + padding
        max_y = (row+1) * self.cell_size - padding
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return None
        return row, col

    def _draw_board(self):
        pygame.draw.rect(self.game_surface, WHITE_COLOR, (0, 0,
                         self.game_surface.get_width(), self.game_surface.get_height()))
        for row in range(8):
            top_offset = row*self.cell_size
            for col in range(8):
                left_offset = col * self.cell_size
                if (row+col) % 2 == 1:
                    pygame.draw.rect(
                        self.game_surface, RED_COLOR, (top_offset, left_offset, self.cell_size, self.cell_size))

        obs = self.env._state.to_obs()
        
        players_colors = [BLUE_COLOR,BLACK_COLOR] 
        for row in range(8):
            top_offset = row*self.cell_size
            for col in range(8):
                left_offset = col * self.cell_size
                if self.player == 1:
                    relative_row = 8-row-1
                    relative_col = 8-col-1
                else:
                    relative_row = row
                    relative_col = col
                if obs[0, relative_row, relative_col] == 1:
                    player_color = players_colors[self.player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                elif obs[1, relative_row, relative_col] == 1:
                    # TODO draw king
                    center = (left_offset+self.cell_size//2, top_offset+self.cell_size//2)                    
                    player_color = players_colors[self.player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                    self.game_surface.blit(self._crown,(center[0]-self._crown.get_width()//2,center[1]-self._crown.get_height()//2))
                elif obs[2, relative_row, relative_col] == 1:
                    player_color = players_colors[1-self.player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                elif obs[3, relative_row, relative_col] == 1:
                    # TODO draw king
                    center = (left_offset+self.cell_size//2, top_offset+self.cell_size//2)  
                    player_color = players_colors[1-self.player]
                    pygame.draw.circle(self.game_surface, GRAY_COLOR, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10+0.1)
                    pygame.draw.circle(self.game_surface, player_color, (
                        left_offset+self.cell_size//2, top_offset+self.cell_size//2), self.cell_size*3//10)
                    self.game_surface.blit(self._crown,(center[0]-self._crown.get_width()//2,center[1]-self._crown.get_height()//2))

    def _encode_action(self, from_pos: tuple[int, int], to_pos: tuple[int, int]):
        f = from_pos
        to = to_pos
        if f is None or to is None:
            return None
        f_row, f_col = f
        to_row, to_col = to
        if self.player != 0:
            # flip board
            f_row, f_col = 8-f_row-1, 8-f_col-1
            to_row, to_col = 8-to_row-1, 8-to_col-1

        row_dir = to_row - f_row
        col_dir = to_col - f_col
        # normalize direction
        if row_dir != 0:
            row_dir /= abs(row_dir)
        if col_dir != 0:
            col_dir /= abs(col_dir)
        direction = (row_dir, col_dir)
        if direction not in EnglishDraughtsState.DIRECTIONS:
            return None
        d = EnglishDraughtsState.DIRECTIONS.index(direction)
        print((f_row,f_col,direction))
        action = self.env._state._encode_action(f_row, f_col, d)
        return action