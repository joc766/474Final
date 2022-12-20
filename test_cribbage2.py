import sys

from policy import CompositePolicy, RandomThrower, RandomPegger, GreedyThrower, GreedyPegger
from cribbage import Game, evaluate_policies
from my_policy import MyPolicy
import time

if __name__ == "__main__":
    games = 2
    if len(sys.argv) > 1:
        games = int(sys.argv[1])
    
    game = Game()
    benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
    submission = MyPolicy(game)
    
    start = time.time()
    results = evaluate_policies(game, submission, benchmark, games)
    end = time.time()
    print("Time: {}".format(end - start))
    print("NET:", results[0])
    print(results)
    
