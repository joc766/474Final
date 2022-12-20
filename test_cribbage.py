import sys

from policy import CompositePolicy, RandomThrower, RandomPegger, GreedyThrower, GreedyPegger
from cribbage import Game, evaluate_policies
from my_policy import MyPolicy
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import concurrent.futures

if __name__ == "__main__":
    games = 2
    if len(sys.argv) > 1:
        games = int(sys.argv[1])
    
    game = Game()
    benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
    submission = MyPolicy(game)
    start = time.time()

    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
    # Submit tasks to the thread pool
        results = [executor.submit(evaluate_policies, game, submission, benchmark, games) for _ in range(cpu_count())]

        # Wait for all tasks to complete
        for result in concurrent.futures.as_completed(results):
            print(f"Result: {result.result()}")

        # Sum the results
        results = [result.result() for result in results]
        sz = len(results)
        total_net = 0
        total_runs = 0
        for r in results:
            total_net += r[0] / sz
            total_runs += games
    end = time.time()
    print("Time: {}".format(end - start))
    print("NET:", total_net)
    print("GAMES: ", total_runs)
    print(results)
    
