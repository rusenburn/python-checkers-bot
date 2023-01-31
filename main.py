import torch as T
from checkers_zero.helpers import get_device
from checkers_zero.networks import SharedResNetwork
from checkers_zero.trainers import AlphaZeroTrainer
from checkers_zero.english_draughts import EnglishDraughtsEnv
from checkers_zero.othello import OthelloGame
from checkers_zero.just_connect4.game import JustConnect4Game
import os
import random


def game_fn():
    return EnglishDraughtsEnv() if random.random() < 0.5 else EnglishDraughtsEnv(all_kings_mode=True)
def test_game_fn():
    return EnglishDraughtsEnv()

def othello_game_fn():
    return OthelloGame()

def justconnect4_game_fn():
    return JustConnect4Game()

def train_justconnect4():
    device = get_device()
    game_fn = justconnect4_game_fn
    game = game_fn()
    network = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    network.to(device=device)

    trainer = AlphaZeroTrainer(
        game_fn=game_fn,
        n_iterations=20,
        n_episodes=128,
        n_sims=50,
        n_epochs=4,
        n_batches=8,
        lr=2.5e-4,
        actor_critic_ratio=0.5,
        n_testing_episodes=20,
        network=network,
        use_async_mcts=True,
        use_mp=True,
        test_game_fn=game_fn
    )
    for nn in trainer.train():
        path = os.path.join("tmp", "justconnect4_async_mp.pt")
        nn.save_model(path)

def train_othello():
    T.set_num_threads(8)
    device = get_device()
    
    game = othello_game_fn()
    network = SharedResNetwork(
        game.observation_space, game.n_actions, n_blocks=5)
    network.to(device=device)
    trainer = AlphaZeroTrainer(
        game_fn=othello_game_fn,
        n_iterations=20,
        n_episodes=128,
        n_sims=25,
        n_epochs=4,
        n_batches=8,
        lr=2.5e-4,
        actor_critic_ratio=0.5,
        n_testing_episodes=20,
        network=network,
        use_async_mcts=False,
        test_game_fn=othello_game_fn
    )
    for nn in trainer.train():
        path = os.path.join("tmp", "othello.pt")
        nn.save_model(path)

def train_english_draughts():
    T.set_num_threads(8)
    device = get_device()
    
    game = game_fn()
    network = SharedResNetwork(
        game.observation_space, game.n_actions, n_blocks=5,filters=128)
    
    path = os.path.join("tmp","english_draught_alpha_zero_20.pt")
    network.load_model(path)
    network.to(device=device)
    trainer = AlphaZeroTrainer(
        game_fn=game_fn,
        n_iterations=20,
        n_episodes=128,
        n_sims=100,
        n_epochs=4,
        n_batches=8,
        lr=2.5e-4,
        actor_critic_ratio=0.5,
        n_testing_episodes=20,
        network=network,
        use_async_mcts=True,
        use_mp=True,
        test_game_fn=test_game_fn
    )
    for nn in trainer.train():
        path = os.path.join("tmp", "english_draught_alpha_zero.pt")
        nn.save_model(path)
def main():
    # train_justconnect4()
    train_english_draughts()
    # train_othello()

    


if __name__ == "__main__":
    main()