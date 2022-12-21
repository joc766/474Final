import sys

from policy import CompositePolicy, RandomThrower, RandomPegger, GreedyThrower, GreedyPegger
from cribbage import Game, evaluate_policies
from my_policy import MyPolicy
import time

if __name__ == "__main__":
    """ "I'm using a neural network to try and improve the keep() function of MyPolicy.
    On average over 500 games I score 0.21 more points than my opponent. This is not an improvement over
    the previous heuristic policy. The model was trained using the train.py script. Further description 
    is in the README.md file. """
    games = 2
    if len(sys.argv) > 1:
        games = int(sys.argv[1])
    
    game = Game()
    benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
    # create my player. The keep() function of MyPolicy. The application of the model starts at line 44 in my_policy.py
    submission = MyPolicy(game)
    
    start = time.time()
    results = evaluate_policies(game, submission, benchmark, games)
    end = time.time()
    print("Time: {}".format(end - start))
    print("NET:", results[0])
    print(results)
    
