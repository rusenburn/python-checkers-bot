# Experimental Python Checkers bot trained using reinforcement learning

## Introduction:
* This project uses a model-based training that is similar to alpha-zero to teach an agent to play a environments that non-stricly alternating players between turns , mainly english draughts and turkish draughts ( dama )


### Getting started
## Requirements:
* python 3.10.4 environment
* git

## Installation steps
* download or clone this repo
    `git clone https://github.com/rusenburn/python-checkers-bot`
* Install python libraries using requirements.txt file
    `pip install -r requirements.txt` or `pip3 install -r requirements.txt`
* For now you have to create a **tmp** folder inside the project folder

* inside `main.py`
```python
from checkers_zero.trainers import AlphaZeroTrainer
from checkers_zero.english_draughts import EnglishDraughtsEnv
from checkers_zero.helpers import get_device
import random
import os

def game_fn():
    if random.random() < 0.5:
        return EnglishDraughtsEnv(all_kings_mode=False)
    return EnglishDraughtsEnv(all_kings_mode=True)

def test_game_fn():
    return EnglishDraughtsEnv(all_kings_mode=False)

def main():
    game = game_fn()
    network = SharedResNetwork(
            shape=game.observation_space,
            n_actions= game.n_actions,
            n_blocks=5,
            filters=128)
    
    # If you wanna load a previously trained network weights
    # path = os.path.join("tmp","english_draught_alpha_zero.pt")
    # network.load_model(path)

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
        test_game_fn=test_game_fn)

    ## trainer.train returns an network iterator
    for nn in trainer.train(): iterator
        path = os.path.join("tmp", "english_draught_alpha_zero.pt")
        # save model after each iteration
        nn.save_model(path)
```
* run `python main.py` or `python3 main.py`
### Known Issues
* Setup.py still not developed
* No documentation yet
* The trainer takes environment function as a parameter which cannot be a lambda function
* UI is not perfect and should be manually used
* Turkish draughts flying king can 180 degree jump even if is illegal