from deck import Deck
from policy import CribbagePolicy, CompositePolicy, GreedyThrower, GreedyPegger
import random
from scoring import score

class MyPolicy(CribbagePolicy):
    def __init__(self, game):
        self._policy = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        
    def keep(self, hand, scores, am_dealer):
        # TODO consider the scores variable
        # TODO probably shouldn't use two private attrs
        # TODO return ties and start breaking non-arbitrarily
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
            for card in remaining_cards:
                total_score += score(game, keep, card, False)[0] + crib * score(game, throw, card, True)[0]
            return keep, throw, total_score

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
        choice = self.apply_heuristic(am_dealer, maxes)
        keep, throw, _ = choice
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





    

                                    
