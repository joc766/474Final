import random

from tensorflow.keras.models import load_model
import numpy as np

from deck import Deck
from policy import CribbagePolicy, CompositePolicy, GreedyThrower, GreedyPegger
from scoring import score
from train import create_encodings


model = load_model("good_greedy_model.h5")

class MyPolicy(CribbagePolicy):
    def __init__(self, game):
        self._policy = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        
    def keep(self, hand, scores, am_dealer):
        # print(am_dealer)
        """ Returns the (keep, throw) pair selected by this policy's
            keep/throw policy.

            hand -- a list of cards
            scores -- the current scores, with this policy's score first
            am_dealer -- a boolean flag indicating whether the crib
                         belongs to this policy
        """
        game = self._policy._game
        crib = 1 if am_dealer else -1
        def score_split(indices):
            keep = []
            throw = []
            deck = game.deck()
            deck.remove(hand)
            remaining_cards = deck.peek(46)
            for i in range(len(hand)):
                if i in indices:
                    throw.append(hand[i])
                else:
                    keep.append(hand[i])
            total_score = 0
            cards = [(c.rank(), c.suit()) for c in keep]
            input_data = np.array([create_encodings(cards)])
            """ This is the part of my mode than applies the neural network """
            # predict the probability of each possible score
            prediction = model.predict(input_data, verbose=0)[0]
            # the prediction is a list of 21 values, each representing the probability of us scoring -10 to 10 points respectively
            # we want to calculate the expected value of each move (where move is the set of cards we end up keeping)
            expected_value = 0
            for i in range(63):
                expected_value += (i-31) * prediction[i]
            total_score += expected_value
            for card in remaining_cards:
                total_score += (1/46) * (score(game, keep, card, False)[0] + crib * score(game, throw, card, True)[0])
            return keep, throw, total_score, prediction

        throw_indices = game.throw_indices()
        random.shuffle(throw_indices)

        moves = map(lambda i: score_split(i), throw_indices)
        maxes = []
        max_score = None
        for move in moves:
            if max_score is None or move[2] > max_score:
                maxes = [move]
                max_score = move[2]
            elif move[2] == max_score:
                maxes.append(move)
        choice = maxes[0]
        print(f"MY HAND: {choice[0]}")
        print(f"PREDICTION: {', '.join('{}:{:.2f}'.format(i-31, np.round(x,2)) for i, x in enumerate(choice[3]))}")
        keep, throw, _, _ = choice
        return keep, throw 


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
        num_options = len(cards)
        # check for special case where we only have 5's in our hand
        all_fives = True
        for card in cards:
            if card.rank() != 5:
                all_fives = False
                break
        for card in cards:
            score = history.score(self._policy._game, card, 0 if am_dealer else 1)
            card_score = card.rank() if card.rank() < 10 else 10
            if score is not None and (best_score is None or (score > best_score and curr_total + card_score != 21)):
                # never play a 5 on the first move if at all possible
                if all_fives or not(history.is_start_round() and card_score == 5):
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
                    return card
            for move in maxes:
                card = move[0]
                card_rank = card.rank()
                if card_rank > 3 and card_rank != 5 and card_rank <=7:
                    return card
            curr_best_rank = card.rank() if card.rank() < 10 else 10
            if curr_total + curr_best_rank == 15:
                for move in maxes:
                    card = move[0]
                    move_rank = card.rank() if card.rank() < 10 else 10
                    if curr_total + move_rank >= 15:
                        return card
           
            
        return best_card





    

                                    
