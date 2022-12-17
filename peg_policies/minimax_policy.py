class Heuristic:
    ''' A wrapper for a heuristic function that counts how many times the
        heuristic is called.
    '''
    def __init__(self, h):
        ''' Creates a wrapper for the given function.

            h -- a heuristic function that takes a game position and returns its heiristic value,
                 or its actual value if the position is terminal.
        '''
        self.calls = 0
        self.heuristic = h
        self.inf = float("inf") 

        
    def evaluate(self, pos):
        ''' Returns the underlying heuristic applied to the given position.

            pos -- a game position
        '''
        # calls on terminal positions don't count
        if not pos.is_terminal():
            self.calls += 1
        return self.heuristic(pos)

    
    def count_calls(self):
        ''' Returns the number of times this heiristic has been called.
        '''
        return self.calls


def minimax_policy(depth, h):
    def fxn(pos):
        value, move = minimax(pos, depth, h)
        return move
    return fxn


def minimax(pos, depth, h):
    ''' Returns the minimax value of the given position, with the given heuristic function
        applied at the given depth.

        pos -- a game position
        depth -- a nonnegative integer
        h -- a heuristic function that can be applied to pos and all its successors
    '''
    if pos.is_terminal() or depth == 0:
        return (h.evaluate(pos), None)
    else:
        if pos.actor() == 0:
            # max player
            best_value = -h.inf
            best_move = None
            moves = pos.get_actions()
            for move in moves:
                child = pos.successor(move)
                mm, _ = minimax(child, depth - 1, h)
                if mm > best_value:
                    best_value = mm
                    best_move = move
            return (best_value, best_move)
        else:
            # min player
            best_value = h.inf
            best_move = None
            moves = pos.get_actions()
            for move in moves:
                child = pos.successor(move)
                mm, _ = minimax(child, depth - 1, h)
                if mm < best_value:
                    best_value = mm
                    best_move = move
            return (best_value, best_move)
