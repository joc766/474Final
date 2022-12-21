import time
from math import sqrt, log
import random

from game import State

def ln(x):
    return log(x)

class MonteCarloNode:
    def __init__(self, parent):
        self.n_visits = 0
        self.total_reward = 0.0
        self.children = None
        self.parent = parent

    def add_successors(self, successors):
        self.children = successors

    def __repr__(self):
        output = f"{self.n_visits}, {self.total_reward}"
        return output

class MonteCarloNode:
    def __init__(self, parent):
        self.n_visits = 0
        self.total_reward = 0.0
        self.children = None
        self.parent = parent

    def add_successors(self, successors):
        self.children = successors

    def __repr__(self):
        output = f"{self.n_visits}, {self.total_reward}"
        return output


class MonteCarloSearch:

    def __init__(self, root: State):
        self.root = root
        self.tree = {root: MonteCarloNode(None)}

    def ucb_value(self, exploit, t, n_i):
        if n_i == 0:
            return float('inf')
        return exploit + sqrt((2 * ln(t)) / n_i)

    def get_next_node(self, position: State):
        # double check algorithm from class
        best_score = float('-inf')
        best_child = None
        tot_visits = self.tree[position].n_visits - 1

        for child_state in self.tree[position].children:
            child_node = self.tree[child_state]
            if child_node.n_visits == 0:
                return child_state
            mean_rwd = child_node.total_reward / child_node.n_visits
            if position.actor() == 1:
                mean_rwd *= -1
            child_score = self.ucb_value(mean_rwd, tot_visits, child_node.n_visits)
            if child_score > best_score:
                best_score = child_score
                best_child = child_state
        return best_child

    def explore_tree(self, end_time):

        while time.time() < end_time:
            curr = self.root
            while self.tree[curr].children != None:
                # traverse the tree until you get to a leaf node (using UCB algorithm)
                curr = self.get_next_node(curr)
            # curr is a state 
            if self.tree[curr].n_visits != 0 and not curr.is_terminal():
                successors = [curr.successor(x) for x in curr.get_actions()]
                self.tree[curr].add_successors(successors)
                for s in successors:
                    self.tree[s] = MonteCarloNode(curr)
                curr = random.choice(self.tree[curr].children)
            final_leaf = curr

            # simulate the game from here
            while not curr.is_terminal():
                moves = curr.get_actions()
                curr = curr.successor(random.choice(moves))
            
            # backprop back up the tree
            reward = curr.payoff()

            curr = final_leaf
            while curr is not None:
                curr_node = self.tree[curr]
                curr_node.n_visits += 1
                curr_node.total_reward += reward
                curr = curr_node.parent

        return self.tree

    def get_best_move(self):
        best_score = float('-inf')
        best_move = None
        s_primes = [(x, self.root.successor(x)) for x in self.root.get_actions()]
        for x, s in s_primes:
            curr_score = self.tree[s].total_reward / self.tree[s].n_visits if self.tree[s].n_visits > 0 else 0
            if self.root.actor() == 1:
                curr_score *= -1
            if curr_score > best_score:
                best_score = curr_score
                best_move = x
        return best_move


def mcts_policy(cpu_time: float):

    def create_policy_func(position: State):
        end_time = time.time() + cpu_time
        mct = MonteCarloSearch(position)
        mct.explore_tree(end_time)
        return mct.get_best_move()

    return create_policy_func
    
    