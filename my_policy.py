from deck import Deck
from policy import CribbagePolicy, CompositePolicy, GreedyThrower, GreedyPegger
import random
from scoring import score
from train import create_encodings

from tensorflow.keras.models import load_model
import numpy as np

class MyPolicy(CribbagePolicy):
    def __init__(self, game):
        self._policy = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        
    def keep(self, hand, scores, am_dealer):
        """ Returns the (keep, throw) pair selected by this policy's
            keep/throw policy.

            hand -- a list of cards
            scores -- the current scores, with this policy's score first
            am_dealer -- a boolean flag indicating whether the crib
                         belongs to this policy
        """
        print("HERE")
        # load the tensorflow model
        model = load_model('model.h5')

        max_move = None
        max_ev = float('-inf')

        # all of the possible combination of moves where we keep 4 cards and throw 2 out of our 6 cards
        for indices in self._policy._game.throw_indices():
            throw = [hand[i] for i in indices]
            keep = [hand[i] for i in range(6) if i not in indices]
            # create the input data for the model
            cards = [(c.rank(), c.suit()) for c in keep]
            input_data = np.array([create_encodings(cards)])
            # predict the probability of each possible score
            prediction = model.predict(input_data)[0]
            # the prediction is a list of 21 values, each representing the probability of us scoring -10 to 10 points respectively
            # we want to calculate the expected value of each move (where move is the set of cards we end up keeping)
            expected_value = 0
            for i in range(21):
                expected_value += (i-10) * prediction[i]
            if expected_value > max_ev:
                max_ev = expected_value
                max_move = (keep, throw)
        return max_move

    def apply_heuristic(self, am_dealer, maxes):
        # keep 5s or cards that add up to 5
        if am_dealer:
            max_score = 0
            choice = None
            for move in maxes:
                throws = move[1]
                curr_score = 0
                throws_sum = 0
                for c in throws:
                    if c.rank() == 5:
                        curr_score += 1
                    throws_sum += c.rank()
                if curr_score > max_score:
                    max_score = curr_score
                    choice = move
            return choice if choice is not None else maxes[0]
        else:
            return maxes[0]




    def peg(self, cards, history, scores, am_dealer):
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
            score = history.score(self._policy._game, card, 0 if am_dealer else 1)
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





    

                                    
