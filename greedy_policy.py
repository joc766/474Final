import random

from pegging import Pegging 
from peg_game import State
from policy import GreedyPegger

def peg(position: State):
    """ Returns the card that maximizes the points earned on the next
        play.  Ties are broken uniformly randomly.

        cards -- a list of cards
        history -- the pegging history up to the point to decide what to play
        scores -- the current scores, with this policy's score first
        am_dealer -- a boolean flag indicating whether the crib
                        belongs to this policy
    """
    # shuffle cards to effectively break ties randomly
    game = position._game
    history = position._history
    turn = position._turn
    cards = list(position._cards[turn])
    random.shuffle(cards)

    best_card = None
    best_score = None
    for card in cards:
        score = history.score(game, card, turn)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_card = card
    return best_card


def greedy_policy():
    def policy_func(position: State):
        action = peg(position)
        return action

    return policy_func