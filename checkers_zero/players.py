from abc import ABC, abstractmethod
import time
import numpy as np
from .state import State
from .networks import NNBase
from .mcts import AMCTS,EvaluatorMCTS,Evaluator,DeepNNEvaluator, RandomRollouts
class PlayerBase(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def choose_action(self,state:State)->int:
        '''
        '''
    

class RandomActionPlayer(PlayerBase):
    def __init__(self) -> None:
        super().__init__()
    
    def choose_action(self, state: State) -> int:
        actions_legality = state.get_actions_legality().astype(np.float32)
        probs = actions_legality / actions_legality.sum()
        a = np.random.choice(len(actions_legality),p=probs)
        return a

class AMCTSPlayer(PlayerBase):
    def __init__(self,n_game_actions:int,nnet:NNBase,n_sims:int,duration_in_millis:float,cpuct=1,temperature=0.5) -> None:
        super().__init__()
        self._n_game_actions:int = n_game_actions
        self.nnet = nnet
        self._temperature = temperature
        self._cpuct = cpuct
        self._n_sims = n_sims
        self._sims_duration_in_millis = duration_in_millis
    
    def choose_action(self, state: State) -> int:
        mcts = AMCTS(self._n_game_actions,self.nnet,cpuct=self._cpuct,temperature=self._temperature)
        t_start = time.perf_counter()
        probs = mcts.search(state,n_minimum_sims=self._n_sims,minimum_duration_in_millis=self._sims_duration_in_millis)
        a = np.random.choice(self._n_game_actions,p=probs)
        duration = time.perf_counter() - t_start
        # print(f"sims per second async\t  {self._n_sims/duration:0.2f}")
        return a

class Human(PlayerBase):
    def __init__(self) -> None:
        super().__init__()

    def choose_action(self, state: State) -> int:
        state.render()
        a = int(input('Choose Action \n'))
        return a

class EvaluatorPlayer(PlayerBase):
    def __init__(self, n_game_actions:int, evaluator: Evaluator, n_sims: int,duration_in_millis:int, temperature=0.5) -> None:
        super().__init__()
        # TODO add cpuct
        self.n_game_actions =n_game_actions
        self.evaluator = evaluator
        self.n_sims = n_sims
        self.temperature = temperature
        self.duration_in_millis = duration_in_millis

    def choose_action(self, state: State) -> int:
        # TODO add cpuct
        mcts = EvaluatorMCTS(self.n_game_actions, self.evaluator, 1,temperature=self.temperature)
        t_start = time.perf_counter()
        probs = mcts.search(state,self.n_sims,self.duration_in_millis)
        a = np.random.choice(len(probs), p=probs)
        duration = time.perf_counter() - t_start
        # print(f"sims per second\t  {self.n_sims/duration:0.2f}")
        return a

class NNMCTSPlayer(EvaluatorPlayer):
    def __init__(self, n_game_actions: int, nnet:NNBase,n_sims: int,duration_in_millis:int, temperature=0.5) -> None:
        evaluator = DeepNNEvaluator(nnet)        
        super().__init__(n_game_actions, evaluator, n_sims,duration_in_millis, temperature)

class RandomRolloutPlayer(EvaluatorPlayer):
    def __init__(self, n_game_actions: int, n_sims: int,duration_in_millis:int, temperature=0.5) -> None:
        evaluator = RandomRollouts()
        super().__init__(n_game_actions, evaluator, n_sims, duration_in_millis,temperature)
    
