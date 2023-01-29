from abc import ABC , abstractmethod
from typing import Iterator,Callable,Any
from checkers_zero.helpers import get_device
from checkers_zero.match import Match
from checkers_zero.mcts import AMCTS,MCTSBase,EvaluatorMCTS,Evaluator
from checkers_zero.networks import NNBase
from checkers_zero.players import AMCTSPlayer, PlayerBase,NNMCTSPlayer,DeepNNEvaluator
from .environment import Environment
from .state import State
from .networks import SharedResNetwork
import copy
import numpy as np
import torch as T
import time
import concurrent.futures
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.multiprocessing as mp

class TrainerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self) -> Iterator[NNBase]:
        raise NotImplementedError()



def assign_rewards(examples:list[tuple[State,np.ndarray,np.ndarray|None,int]],rewards:np.ndarray,last_player:int)->list[tuple[State,np.ndarray,np.ndarray]]:
    inverted :np.ndarray = rewards[::-1]
    results = [(ex[0],ex[1],rewards if ex[3]==last_player else inverted) for ex in examples]
    return results



def execute_episode(game_fn:Callable[[],Environment],n_sims:int,network:NNBase,use_amcts:bool):
    examples :list[tuple[State,np.ndarray,np.ndarray|None,int]] = []
    results:list[tuple[State,np.ndarray,np.ndarray]] = []
    network.to(get_device())
    network.eval()
    
    game = game_fn()
    temperature = 1.0
    cpuct = 2.0
    player : MCTSBase = AMCTS(game.n_actions,network,cpuct,temperature) if use_amcts else EvaluatorMCTS(
        game.n_actions,DeepNNEvaluator(network),cpuct,temperature)

    state = game.reset()
    while True:
        current_player = state.player_turn()
        probs = player.search(state,n_minimum_sims=n_sims,minimum_duration_in_millis=0)
        examples.append((state,probs,None,current_player))
        syms = state.get_symmetries(probs)
        for s,p in syms:
            examples.append((s,p,None,current_player))
        
        action = np.random.choice(len(probs),p=probs)
        state = state.step(action)
        if state.is_terminal():
            results = assign_rewards(examples,state.game_result(),state.player_turn())
            break
    return results

def execute_episode_process(args):
    return execute_episode(*args)

def execute_match(game_fn:Callable[[],Environment],network_1:NNBase,network_2:NNBase,n_sims:int,n_sets:int,use_amcts:bool):
    device = get_device()
    network_1.to(device=device)
    network_2.to(device=device)
    network_1.eval()
    network_2.eval()
    n_game_actions = game_fn().n_actions
    temperature = 0.5
    cpuct = 1
    player_1  = AMCTSPlayer(n_game_actions,network_1,n_sims=n_sims,duration_in_millis= 0,cpuct=cpuct,temperature=temperature) if use_amcts else NNMCTSPlayer(
        n_game_actions,network_1,n_sims,0,temperature)
    player_2 = AMCTSPlayer(n_game_actions,network_2,n_sims=n_sims,duration_in_millis= 0,cpuct=cpuct,temperature=temperature) if use_amcts else NNMCTSPlayer(
        n_game_actions,network_2,n_sims,0,temperature
    )
    match_ = Match(game_fn=game_fn,player_1=player_1,player_2=player_2,n_sets= n_sets,render=False)
    wdl = match_.start()
    return wdl

def execute_match_process(args):
    return execute_match(*args)

class AlphaZeroTrainer(TrainerBase):
    def __init__(self,game_fn:Callable[[],Environment],
            test_game_fn:Callable[[],Environment],
            n_iterations:int,
            n_episodes:int,
            n_sims:int,
            n_epochs:int,
            n_batches:int,
            lr:float,
            actor_critic_ratio:float,
            n_testing_episodes:int,
            network:NNBase,
            use_async_mcts=True,
            checkpoint:str|None=None) -> None:
        super().__init__()
        self._game_fn = game_fn
        self._test_game_fn = test_game_fn
        self._n_iterations = n_iterations
        self._n_episodes = n_episodes
        self._n_sims = n_sims
        self._lr = lr
        self._actor_critic_ratio = actor_critic_ratio
        self._n_epochs = n_epochs
        self._n_batches = n_batches
        self._n_testing_episodes = n_testing_episodes
        self._use_async_mcts = use_async_mcts
        self._checkpoint:str|None = checkpoint
        game = game_fn()
        self._n_game_actions = game.n_actions
        self.base_network : NNBase = SharedResNetwork(game.observation_space,n_actions=game.n_actions,n_blocks=5) if network is None else network
        self.base_network.to(get_device())
    
    def train(self)->Iterator[NNBase]:
        strongest_network = copy.deepcopy(self.base_network)
        strongest_network.to(get_device())
        examples :list[tuple[State,np.ndarray,np.ndarray]] = []
        for iteration in range(self._n_iterations):
            t_collecting_start = time.perf_counter()
            print(f"Iteration {iteration+1} of {self._n_iterations}")
            n_workers = 8
            examples = []
            # self.base_network.cpu()
            # n_processes = 3
            # with mp.Pool(processes=n_processes) as pool:
            #     a = pool.map(execute_episode_process,tqdm(list(zip([self._game_fn for _ in range(self._n_episodes)],[self._n_sims for _ in range(self._n_episodes)],[self.base_network for _ in range(self._n_episodes)],[self._use_async_mcts for _ in range(self._n_episodes)]))))
            #     examples+=[y for x in tqdm(a,desc="Collecting Data") for y in x]
            # self.base_network.to(get_device())
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                a = executor.map(execute_episode,
                    [self._game_fn for _ in range(self._n_episodes)],
                    [self._n_sims for _ in range(self._n_episodes)],
                    [self.base_network for _ in range(self._n_episodes)],
                    [self._use_async_mcts for _ in range(self._n_episodes)])
                examples+=[y for x in tqdm(a,desc="Collecting Data") for y in x]

            states,probs,wdl = list(zip(*examples))
            states = [ex[0] for ex in examples]
            probs = [ex[1] for ex in examples]
            wdl = [ex[2] for ex in examples]

            obs = [state.to_obs() for state in states]
            collecting_data_duration = time.perf_counter() - t_collecting_start


            t_training_start = time.perf_counter()
            n_examples :int =  len(examples)
            print(f"Training Phase using {n_examples} examples...")
            self._train_network(self.base_network,obs,probs,wdl)
            training_duration = time.perf_counter() - t_training_start
            self.base_network.eval()

            t_evaluation_start = time.perf_counter()
            if iteration % 5 == 4 or iteration == 0 or iteration==self._n_iterations:
                print("Evaluating")
                n_processes = 3
                # n_sets_per_process = int(self._n_testing_episodes // n_processes)
                wdl = np.zeros((3,))
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exec:
                    a = exec.map(execute_match,
                        [self._test_game_fn for _ in range(self._n_testing_episodes//2)],
                        [self.base_network for _ in range(self._n_testing_episodes//2)],
                        [strongest_network for _ in range(self._n_testing_episodes//2)],
                        [self._n_sims for _ in range(self._n_testing_episodes//2)],
                        [2 for _ in range(self._n_testing_episodes//2)],
                        [self._use_async_mcts for _ in range(self._n_testing_episodes//2)])
                    for res in a:
                        wdl+=res
                # with mp.Pool(processes=n_processes) as pool:
                #     a = pool.map(execute_match_process,tqdm(list(zip([self._test_game_fn for _ in range(n_processes)],[self.base_network for _ in range(n_processes)],[strongest_network for _ in range(n_processes)],[self._n_sims for _ in range(n_processes)],[n_sets_per_process for _ in range(n_processes)],[self._use_async_mcts for _ in range(n_processes)]))))
                #     for res in a:
                #         wdl += res
                    
                print("Evaluation Phase")

                score_ratio = (wdl[0]*2 + wdl[1]) / (wdl.sum()*2)
                print(f"wins : {wdl[0]} , draws:{wdl[1]} , losses:{wdl[2]}")

                print(
                    f"score ratio against old strongest opponent: {score_ratio*100:0.2f}%")
                if wdl[0] > wdl[2]:
                    strongest_network.load_state_dict(
                        self.base_network.state_dict())
            evaluation_duration = time.perf_counter() - t_evaluation_start
            iteration_duration = time.perf_counter() - t_collecting_start
            print("**************************************************************")
            print(
                f"Iteration                  {iteration+1} of {self._n_iterations}")
            print(
                f"Iteration Duration         {iteration_duration:0.2f} seconds")
            print(
                f"Collecting Data Duration   {collecting_data_duration:0.2f} seconds")
            print(
                f"Training Duration          {training_duration:0.2f} seconds")
            print(
                f"Evaluation Duration        {evaluation_duration:0.2f} seconds")
            print(
                f"Training Data Count        {n_examples} examples")
            print("**************************************************************")

            yield strongest_network
        return strongest_network


    def _train_network(self,nnet:NNBase,obs:list[np.ndarray],probs:list[np.ndarray],wdl:list[np.ndarray]):
        device = get_device()
        nnet.train()
        optimizer =  T.optim.Adam(nnet.parameters(),self._lr,weight_decay=1e-4)

        obs_ar = np.array(obs)
        probs_ar = np.array(probs)
        wdl_ar = np.array(wdl)
        batch_size = int(len(obs)//self._n_batches)
        for epoch in range(self._n_epochs):
            sample_ids :np.ndarray
            for _ in range(self._n_batches):
                sample_ids = np.random.randint(len(obs),size=batch_size)
                obs_batch: np.ndarray = obs_ar[sample_ids]
                probs_batch : np.ndarray = probs_ar[sample_ids]
                wdl_batch:np.ndarray = wdl_ar[sample_ids]

                obs_t = T.tensor(obs_batch,dtype=T.float32,device=device)
                target_probs = T.tensor(probs_batch,dtype=T.float32,device=device)
                target_wdl = T.tensor(wdl_batch,dtype=T.float32,device=device)
                predicted_probs :T.Tensor
                predicted_wdl:T.Tensor
                predicted_probs,predicted_wdl = nnet(obs_t)
                actor_loss = self._cross_entropy_loss(target_probs,predicted_probs)
                critic_loss = self._cross_entropy_loss(target_wdl,predicted_wdl)
                total_loss:T.Tensor = actor_loss + self._actor_critic_ratio * critic_loss
                optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(nnet.parameters(),0.5)
                optimizer.step()
        if T.cuda.is_available():
            T.cuda.empty_cache()

    @staticmethod
    def _cross_entropy_loss(target_probs: T.Tensor, predicted_probs: T.Tensor):
        log_probs = predicted_probs.log()
        loss = -(target_probs*log_probs).sum(dim=-1).mean()
        return loss





    




