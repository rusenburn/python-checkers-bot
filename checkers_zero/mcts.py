from abc import ABC,abstractmethod
import numpy as np
import time
import math
import torch as T
from checkers_zero.helpers import get_device
from checkers_zero.networks import NNBase
from .state import State


EPS = 1e-8
MAX_ASYNC_SIMULATIONS = 4
DEFAULT_N = 1
DEFAULT_W = -1

class MCTSBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def search(self,state:State,n_minimum_sims:int,minimum_duration_in_millis:float)->np.ndarray:
        raise NotImplementedError()

class Evaluator(ABC):
    '''
    abstract class for monte carlo state evaluation
    '''
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def evaluate(self,state:State)->tuple[np.ndarray,np.ndarray]:
        '''
        Takes a state and returns a tuple of two numpy arrays
        which represents policy actions probs and state evaluation
        '''
class DeepNNEvaluator(Evaluator):
    def __init__(self,nnet:NNBase) -> None:
        super().__init__()
        self.nnet = nnet
        self.device = get_device()
        # self.nnet.eval()
        # self.nnet.to(self.device)
    
    def evaluate(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        assert not state.is_terminal()
        obs = state.to_obs()
        obs_t = T.tensor(np.array([obs]),dtype=T.float32,device=self.device)
        with T.no_grad():
            res :tuple[T.Tensor,T.Tensor]= self.nnet(obs_t)
            probs_t ,wdl_t= res
        probs_ar = probs_t.cpu().numpy()[0]
        wdl_ar = wdl_t.cpu().numpy()[0]
        return probs_ar,wdl_ar


class RandomRollouts(Evaluator):
    '''
    Evaluating state 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        '''
        Evaluating state by playing random legal moves until the game ends
        and return the score , and returns equally probs over legal moves
        returns probs , win-draw-loss score
        '''
        assert not state.is_terminal()
        starting_player = state.player_turn()
        actions_legality =  state.get_actions_legality()
        probs = actions_legality.astype(np.float32) / actions_legality.sum()
        while not state.is_terminal():
            actions_legality =  state.get_actions_legality()
            best_actions = np.array(np.argwhere(actions_legality == 1).flatten())
            best_action = np.random.choice(best_actions)
            best_a = best_action
            state = state.step(best_a)
        wdl = state.game_result()
        last_player = state.player_turn()
        if last_player != starting_player:
            wdl = wdl[::-1]
        return probs,wdl
        
class AMCTS(MCTSBase):
    def __init__(self,n_game_actions:int,nnet:NNBase,cpuct:float,temperature:float) -> None:
        super().__init__()
        self._n_game_actions = n_game_actions
        self._nnet = nnet
        self._cpuct = cpuct
        self._temperature = temperature
        self._states :set[tuple]= set()
        self._edges:dict[tuple,list[State|None]] = dict()
        self._ns : dict[tuple,int]= dict()
        self._nsa : dict[tuple,np.ndarray] = dict()
        self._wsa : dict[tuple,np.ndarray] = dict()
        self._psa : dict[tuple,np.ndarray] = dict()
        self._actions_legality :dict[tuple,np.ndarray]= dict()
        self._root :State|None= None
        self._root_player :int | None = None

        # saves all the states to be evaluated in roll method along the path to backpropogate score
        self._rollouts:list[tuple[State,list[tuple[tuple,int,int]]]] = []

    def search(self, state: State,n_minimum_sims:int,minimum_duration_in_millis:float) -> np.ndarray:
        self._root = state
        return self._search_root(n_minimum_sims,minimum_duration_in_millis)
    
    def _search_root(self,n_minimum_sims:int,min_duration_in_millis:float)->np.ndarray:
        assert self._root is not None
        self._root_player = self._root.player_turn()
        actions_legality = self._root.get_actions_legality()
        if actions_legality.sum() == 1:
            return actions_legality.astype(np.float32).copy()
        duration_in_seconds = min_duration_in_millis / 1000
        t_start = time.perf_counter()
        t_end = t_start + duration_in_seconds
        sim_count = 0
        while sim_count < n_minimum_sims:
            self._simulate_one(self._root)
            if sim_count% MAX_ASYNC_SIMULATIONS == MAX_ASYNC_SIMULATIONS-1:
                self._roll()
            sim_count+=1
        
        while t_end > time.perf_counter():
            self._simulate_one(self._root)
            sim_count+=1
            if sim_count% MAX_ASYNC_SIMULATIONS == MAX_ASYNC_SIMULATIONS-1:
                self._roll()
        
        self._roll()
        return self._get_probs()
    def _simulate_one(self,state:State,visited_path:list[tuple[tuple,int,int]]|None = None):
        if visited_path is None:
            visited_path = []
        if state.is_terminal():
            wdl = state.game_result()
            player = state.player_turn()
            # wins - losses
            z = wdl[0] - wdl[2]
            self._backprop(visited_path,z,player)
            return
        short:tuple = state.to_short()
        player= state.player_turn()
        if short not in self._states:
            self._expand_state(state,short)
            actions_legality = self._actions_legality[short]
            if actions_legality.sum() == 1: # only 1 legal action , change probs to have this action as 1 and the rest are 0 , skip rollout for this state
                self._psa[short] = actions_legality.astype(np.float32)
            else:
                self._add_to_rollouts(state,visited_path)
                return
        
        best_action :int = self._find_best_action(short)

        if self._edges[short][best_action] is None:
            self._edges[short][best_action] = state.step(best_action)
        
        new_state : State|None = self._edges[short][best_action]

        if new_state is None:
            raise ValueError()
        
        visited_path.append((short,best_action,player))
        self._simulate_one(new_state,visited_path)
        self._nsa[short][best_action] +=DEFAULT_N
        self._ns[short]+=DEFAULT_N
        self._wsa[short][best_action] += DEFAULT_W

    
    def _find_best_action(self,short:tuple)->int:
        max_u , best_action = -float("inf"),-1
        wsa_ar = self._wsa[short]
        nsa_ar = self._nsa[short]
        psa_ar = self._psa[short]
        ns = self._ns[short]
        actions_legality = self._actions_legality[short]
        for action,is_legal in enumerate(actions_legality):
            if not is_legal:
                continue
            psa : float = psa_ar[action]
            nsa : float = nsa_ar[action]
            qsa : float = 0.0
            if nsa > 0:
                qsa = wsa_ar[action] / (nsa+EPS)
            u = qsa + self._cpuct * psa * math.sqrt(ns)/(1+nsa)
            if u > max_u:
                max_u = u
                best_action = action
        if best_action == -1:
            best_actions = np.array(np.argwhere(
                actions_legality == 1).flatten())
            best_action = np.random.choice(best_actions)
        return best_action

    def find_ba(self,short:tuple)->int:
        wsa_ar = self._wsa[short].astype(np.float32)
        nsa_ar = self._nsa[short].astype(np.float32)
        psa_ar = self._psa[short]
        ns = self._ns[short]
        actions_legality = self._actions_legality[short]

        qsa_ar = wsa_ar/(nsa_ar+EPS)
        u_ar:np.ndarray = qsa_ar + self._cpuct * psa_ar * math.sqrt(ns) / (1+nsa_ar)
        u_ar -= (actions_legality == 0)*-1000000
        action = int(u_ar.argmax(axis=None))
        return action

    def _backprop(self,visited_path:list[tuple[tuple,int,int]],last_score:float,last_player:int):
        while len(visited_path)!=0:
            short ,action,player = visited_path.pop()
            score = last_score if player == last_player else -last_score
            self._ns[short]+=1-DEFAULT_N
            self._nsa[short][action]+=1-DEFAULT_N
            self._wsa[short][action]+=score-DEFAULT_W
    
    def _expand_state(self,state:State,short:tuple):
        assert not state.is_terminal()
        self._states.add(short)
        actions_legality = state.get_actions_legality()
        self._actions_legality[short] = actions_legality
        self._ns[short] = 0
        self._nsa[short] = np.zeros((self._n_game_actions,),dtype=np.float32)
        self._wsa[short] = np.zeros((self._n_game_actions,),dtype=np.float32)
        self._edges[short] = [None for _ in range(self._n_game_actions)]
        a = actions_legality.astype(dtype=np.float32)
        psa : np.ndarray = a/a.sum(keepdims=True)
        self._psa[short] = psa

    def _add_to_rollouts(self,state:State,visited_path:list[tuple[tuple,int,int]]):
        self._rollouts.append((state,visited_path))
        
    
    def _get_probs(self)->np.ndarray:
        assert self._root is not None
        short = self._root.to_short()
        actions_visits = self._nsa[short]
        temperature = self._temperature
        if temperature == 0:
            max_action_visits = np.max(actions_visits)
            best_actions = np.array(np.argwhere(
                actions_visits == max_action_visits)).flatten()
            best_action = np.random.choice(best_actions)

            probs: np.ndarray = np.zeros(
                (len(actions_visits),), dtype=np.float32)
            probs[best_action] = 1
            return probs
        
        probs_with_temperature = actions_visits.astype(
            np.float32)**(1/temperature)
        probs = probs_with_temperature/probs_with_temperature.sum()
        return probs
    
    def _roll(self):
        if len(self._rollouts) == 0:
            return
        # self._nnet.eval()
        states :list[State] = [
            tup[0] for tup in self._rollouts
        ]

        obs_ar = np.array([s.to_obs() for s in states],dtype=np.float32)
        obs_t = T.tensor(obs_ar,dtype=T.float32,device=get_device())
        with T.no_grad():
            dis : tuple[T.Tensor,T.Tensor] = self._nnet(obs_t)
            probs_t , wdl_t = dis
        wdl_ar :np.ndarray=  wdl_t.cpu().numpy()

        for i, wdl in enumerate(wdl_ar):
            visited_path = self._rollouts[i][1]
            z = wdl[0]-wdl[2]
            self._backprop(visited_path,z,states[i].player_turn())
        
        probs_ar:np.ndarray = probs_t.cpu().numpy()
        prob:np.ndarray
        for i,prob in enumerate(probs_ar):
            state = self._rollouts[i][0]
            short = state.to_short()
            actions_legality = self._actions_legality[short]
            assert not np.any(actions_legality!=state.get_actions_legality())
            prob = prob*actions_legality
            assert prob.sum() > 0
            prob = prob / prob.sum(keepdims=True)
            self._psa[short]  = prob

        self._rollouts.clear()


class EvaluatorMCTS(MCTSBase):
    def __init__(self,n_game_actions:int,evaluator:Evaluator,cpuct:float,temperature:float) -> None:
        super().__init__()
        self._n_game_actions = n_game_actions
        self._evaluator = evaluator
        self._cpuct = cpuct
        self._temperature = temperature
        self._root_node =None
    
    def search(self, state: State, n_minimum_sims: int, minimum_duration_in_millis: int) -> np.ndarray:
        assert not state.is_terminal()
        self._root_node = Node(state,self._n_game_actions,self._cpuct)
        return self._root_node.search_and_get_probs(self._evaluator,n_sims=n_minimum_sims,duration_in_millis=minimum_duration_in_millis,temperature=self._temperature)


class Node:
    def __init__(self,state:State,n_game_actions:int,cpuct:float=1) -> None:
        self.state = state
        self.actions_legality : np.ndarray|None= None
        self.cpuct = cpuct
        self.n_game_actions = n_game_actions
        self.children : list[Node|None] = [None] * n_game_actions
        self.probs :np.ndarray= np.zeros((n_game_actions,),dtype=np.float32)
        self.n:int = 0
        self.na:np.ndarray = np.zeros((n_game_actions,),dtype=np.int32)
        self.wdla:np.ndarray = np.zeros((n_game_actions,3),dtype=np.float32)
        self.wa : np.ndarray = self.wdla[:,0]
        self.da : np.ndarray = self.wdla[:,1]
        self.la : np.ndarray = self.wdla[:,2]

        self.is_terminal:bool|None = None
        self.game_result :np.ndarray|None= None


    def search(self,evaluator:Evaluator)->tuple[np.ndarray,int]:
        if self.is_terminal is None:
            self.is_terminal = self.state.is_terminal()
        
        if self.is_terminal:
            if self.game_result is None:
                self.game_result = self.state.game_result()
            
            # TODO DONE
            return self.game_result,self.state.player_turn()
        if self.actions_legality is None:
            self.actions_legality =  self.state.get_actions_legality()
            self.children = [None for _ in self.actions_legality]
            probs , wdl = evaluator.evaluate(self.state)
            self.probs = probs
            # TODO DONE
            return wdl,self.state.player_turn()
        
        a:int = self._get_best_action()

        if self.children[a] is None:
            new_state = self.state.step(a)
            self.children[a] = Node(new_state,self.n_game_actions,self.cpuct)
        
        new_node = self.children[a]
        assert new_node is not None

        # TODO DONE
        wdl,player  = new_node.search(evaluator)

        if self.state.player_turn() != player:
            wdl = wdl[::-1]
        self.wa[a]+=wdl[0]
        self.da[a]+=wdl[1]
        self.la[a]+=wdl[2]
        self.na[a]+=1
        self.n+=1

        # TODO DONE
        return wdl,self.state.player_turn()
    
    def get_probs(self,temperature)->np.ndarray:
        # get a copy of array of number of times an action was performed in this node during search
        action_visits :np.ndarray = self.na.copy()
        # if exploring temperature was 0 , give the best action 
        if temperature == 0:
            max_action_visits = np.max(action_visits)
            best_actions = np.array(np.argwhere(action_visits == max_action_visits)).flatten()
            best_action = np.random.choice(best_actions)    
            
            probs : np.ndarray = np.zeros((len(action_visits),),dtype=np.float32)
            probs[best_action] = 1
            return probs

        # if exploring temperature was not 0 get an action depends on the number of times this action was perfomed 
        probs_with_temperature:np.ndarray = action_visits.astype(np.float32)**(1/temperature)
        probs = probs_with_temperature/probs_with_temperature.sum()
        return probs

    def search_and_get_probs(self,evaluator:Evaluator,n_sims:int,duration_in_millis:int,temperature:float)->np.ndarray:
        if self.is_terminal is None:
            self.is_terminal = self.state.is_terminal()
            assert not self.is_terminal
        t_end = time.perf_counter() + duration_in_millis/1000
        for _ in range(n_sims):
            self.search(evaluator)
        
        while t_end > time.perf_counter():
            self.search(evaluator)
        probs = self.get_probs(temperature)
        return probs
    
    def _get_best_action(self):
        assert self.actions_legality is not None
        max_u,best_a = -float("inf"),-1
        for a , is_legal in enumerate(self.actions_legality):
            if not is_legal:
                continue
            
            na:int = self.na[a]
            qsa : float = 0
            if na > 0 :
                wa = self.wa[a]
                la = self.la[a]
                qsa = (wa - la) / na
            u = qsa + self.cpuct * self.probs[a] * math.sqrt(self.n) / (1 + na)

            # u = qsa + self.cpuct * self.ps[s][a] * math.sqrt(self.ns[s]) / (1 + nsa)
            if u > max_u:
                max_u = u
                best_a = a
        if best_a == -1:
            # should not happen , unless was given a very bad probabilities by nnet
            # pick random action from legal actions
            best_actions = np.array(np.argwhere(self.actions_legality == 1).flatten())
            best_action = np.random.choice(best_actions)
            best_a = best_action
        
        a = best_a
        return a
