import random

from pegging import Pegging 
from peg_game import State

def peg(position: State):
    game = position._game
    history = position._history
    turn = position._turn
    cards = list(position._cards[turn])
    random.shuffle(cards)
    prev_card = history._card
    curr_total = history._total

    # find all the pairs
    seen = set()
    pairs = set()
    for c in cards:
        if c.rank() in seen:
            pairs.add(c.rank())
        seen.add(c.rank())


    best_card = None
    best_score = None
    maxes = []
    for card in cards:
        score = history.score(game, card, turn)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            maxes = [(card, score)]
            best_card = card
        elif score is not None and score == best_score:
            maxes.append((card, score))
    if len(maxes) > 1:
        for move in maxes:
            card = move[0]
            card_rank = card.rank()
            if card_rank in pairs:
                # print("PAIR INFO: ")
                # print(pairs)
                # print(cards)
                # print(card_rank)
                best_card = card
    return best_card

def heuristic_policy():
    def policy_func(position: State):
        action = peg(position)
        return action
    
    return policy_func
        