import numpy as np
from abc import ABC,abstractmethod

class State(ABC):
    def __init__(self)->None:
        super().__init__()
    
    @property
    @abstractmethod
    def n_actions(self)->int:
        '''
        returns the number of game actions
        '''
    
    @property
    @abstractmethod
    def observation_space(self)->tuple:
        '''
        returns the shape of observation
        '''
    @abstractmethod
    def get_actions_legality(self)->np.ndarray:
        '''
        Returns binary numpy array with length of the number of actions, 
        legal actions have a value of 1
        ex: if legal_action is [1,0,0,0,1] then legal_action[0] = 1 legal ,
        but legal_action[1] is not
        '''
    
    @abstractmethod
    def is_terminal(self)->bool:
        '''
        Returns True if the game is over
        or False if it is not
        '''
    
    @abstractmethod
    def game_result(self)->np.ndarray:
        '''
        Returns a numpy array of win-draw-loss
        '''
    
    @abstractmethod
    def step(self,action:int)->'State':
        '''
        Returns the new state of the game after
        peforming an action
        '''
    
    @abstractmethod
    def to_obs(self)->np.ndarray:
        '''
        Converts the state into numpy array and returns it
        '''
    
    @abstractmethod
    def player_turn(self)->int:
        '''
        return which player supposed to play
        '''
    @abstractmethod
    def render(self)->None:
        '''
        Renders the current state
        '''
    
    @abstractmethod
    def to_short(self)->tuple:
        '''
        Returns a short represntation of the current state
        '''
    @abstractmethod
    def get_symmetries(self,probs:np.ndarray)->list[tuple['State',np.ndarray]]:
        '''
        Takes State action or action probs and returns 
        List of equivalent states with provided probs
        '''

