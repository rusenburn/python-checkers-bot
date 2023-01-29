import numpy as np
import tqdm
from .players import PlayerBase
from .environment import Environment
from .state import State
from typing import Callable

class Match():
    def __init__(self, game_fn: Callable[[],Environment], player_1: PlayerBase, player_2: PlayerBase, n_sets=1, render=False,verbose=False) -> None:
        self.player_1 = player_1
        self.player_2 = player_2
        self.game_fn = game_fn
        self.game = game_fn()
        self.n_sets = n_sets
        self.render = render
        self.verbose = verbose
        self.scores = np.zeros((3,), dtype=np.int32)  # W - D - L for player_1

    def start(self) -> np.ndarray:
        starting_player = 0
        s = self.game.reset()
        
        for _ in tqdm.tqdm(range(self.n_sets),desc="Versus") if self.verbose else range(self.n_sets):
            scores = self._play_set(starting_player)
            self.scores += scores
            starting_player = 1-starting_player
        return self.scores


    def start_2(self)->np.ndarray:
        state = self.game.reset()
        for _ in tqdm.tqdm(range(self.n_sets),desc="Versus") if self.verbose else range(self.n_sets):
            scores = self._play_set_2()
            self.scores+=scores
        return self.scores.copy()

    def _play_set_2(self)->np.ndarray:
        players = [self.player_1,self.player_2]
        state = self.game.reset()
        while not state.is_terminal():
            if self.render:
                state.render()
            current_player = state.player_turn()
            player = players[current_player]
            action = player.choose_action(state)
            state = state.step(action)
        
        wdl = state.game_result()
        player = state.player_turn()
        if player != 0:
            wdl = wdl[::-1]
        if self.render:
            state.render()
        return wdl.copy()
        
            

    def _play_set(self, starting_player:int) -> np.ndarray:
        players = [self.player_1, self.player_2]
        state = self.game.reset()
        done = False
        game_p = state.player_turn()
        inverted = False if game_p == starting_player else True
        # current_player = starting_player
        current_player = 1-game_p if inverted else game_p
        while True:
            if self.render:
                state.render()
            player = players[current_player]
            a = player.choose_action(state)
            legal_actions = state.get_actions_legality()
            if not legal_actions[a]:
                print(f'player {current_player+1} chose wrong action {a}\n')
                continue
            new_state: State
            new_state, done = self.game.step(a)
            state = new_state
            game_p = state.player_turn()
            current_player = 1-game_p if inverted else game_p
            # current_player = 1-current_player
            if done:
                assert new_state.is_terminal()
                result: np.ndarray = new_state.game_result()
                if current_player != 0:
                    result = result[::-1]
                break
        if self.render:
            state.render()
        return result