import argparse

from peg_game import PeggingGame
from test_policies import create_agent
from deck import Card

def simulate(game: PeggingGame, p1, p2):
    # start with fresh copies of the policy functions
    p1_policy = p1()
    p2_policy = p2()
    position = game.initial_state()
    p1_cards = [(x.rank(), x.suit()) for x in position._cards[0]]
    # copy = position
    
    while not position.is_terminal():
        if position.actor() == 0:
            move = p1_policy(position)
        else:
            move = p2_policy(position)
        position = position.successor(move)

    #checking that minimax is working correctly by testing on pegging
    # and ensuring that MCTS never beats minimax with depth 14, which can search the entire
    # tree and so is optimal
    #while not copy.is_terminal():
    #    move = p2_policy(copy)
    #    copy = copy.successor(move)
    #if (i % 2 == 0 and position.payoff() > copy.payoff()) or (i % 2 == 1 and position.payoff() < copy.payoff()):
    #    print("COPY: " + str(copy))
        
    # to see final position, which for pegging includes the
    # complete sequence of cards played
    # print(position)

    p1_score = position.payoff() # p1 score - p2 score 
    return p1_cards, p1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MCTS agent")
    parser.add_argument("--agent1", dest="agent1", choices=["mcts", "minimax", "greedy", "heuristic"], default="mcts", help="agent to test (default=mcts)")
    parser.add_argument("--agent2", dest="agent2", choices=["mcts", "minimax", "greedy", "heuristic"], default="minimax", help="agent to test (default=mcts)")
    args = parser.parse_args()

    game = PeggingGame(4)
    agent1 = create_agent(args.agent1)
    agent2 = create_agent(args.agent2)

    print(simulate(game, agent1, agent2))