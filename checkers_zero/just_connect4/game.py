import numpy as np
from checkers_zero.environment import Environment
from .state import N_COLS,N_ROWS,NO_LAST_ACTION,JustConnect4State

class JustConnect4Game(Environment):
    def __init__(self) -> None:
        super().__init__()
        self._state = self._initialize_new_state()

    @property
    def observation_space(self) -> tuple:
        return self._state.observation_space
    
    @property
    def n_actions(self) -> int:
        return self._state.n_actions
    
    def reset(self) -> 'JustConnect4State':
        self._state = self._initialize_new_state()
        return self._state
    
    def render(self) -> None:
        self._state.render()

    def step(self, action: int) -> tuple[JustConnect4State, bool]:
        new_state  = self._state.step(action)
        done = new_state.is_terminal()
        self._state = new_state
        return new_state,done
        
    def _initialize_new_state(self)->'JustConnect4State':
        obs = np.zeros((4,N_ROWS,N_COLS),dtype=np.int32)
        turn = 0
        last_action = NO_LAST_ACTION
        return JustConnect4State(obs,turn,last_action)
    