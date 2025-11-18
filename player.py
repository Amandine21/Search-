import random
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


MAX_DEPTH = 12
TIME_LIMIT = 0.055      


class PlayerControllerMinimax(PlayerController):
    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            node = Node(message=msg, player=0)

            best_move = self.search_best_next_move(node)
            self.sender({"action": best_move, "search_time": None})

    # -----------------------------------------------------------
    # HASHING
    def hash_key(self, state):

        pos_dic = dict()
        for pos, score in zip(state.get_fish_positions().items(),
                              state.get_fish_scores().items()):
            score = score[1]
            pos = pos[1]
            k = f"{pos[0]}{pos[1]}"
            pos_dic[k] = score

        return str(state.get_hook_positions()) + str(pos_dic)


    # -----------------------------------------------------------
    # HEURISTIC EVALUATION
    def evaluate_node(self, node):
        state = node.state
        fish_positions = state.get_fish_positions()
        hooks = state.get_hook_positions()
        fish_scores = state.get_fish_scores()

        def torus_manhattan(a, b, width=20):
            dx = abs(a[0] - b[0])
            dx = min(dx, width - dx)
            dy = abs(a[1] - b[1])
            return dx + dy

        max_score = 0
        min_score = 0

        for fish_id, pos in fish_positions.items():
            val = fish_scores[fish_id]

            d_max = torus_manhattan(hooks[0], pos) + 1
            d_min = torus_manhattan(hooks[1], pos) + 1

            if hooks[0] == pos:
                max_score += val
            else:
                max_score += val / d_max

            if hooks[1] == pos:
                min_score += val
            else:
                min_score += val / d_min

        return max_score - min_score


    # -----------------------------------------------------------
    # ALPHA-BETA PRUNING
    def alphabeta(self, node, state, depth, alpha, beta, player,
                  initial_time, seen_nodes):

        # Create a time cutoff
        if time.time() - initial_time > TIME_LIMIT:
            raise TimeoutError

        # Store the states in the transposition table
        k = self.hash_key(state)
        if k in seen_nodes and seen_nodes[k][0] >= depth:
            return seen_nodes[k][1]

        children = node.compute_and_get_children()

        #Reorder the scores in descending order
        children.sort(key=self.evaluate_node, reverse=True)

        # Checks if the current node is a terminal node
        if depth == 0 or not children:
            value = self.evaluate_node(node)

        # MAX PLAYER
        elif player == 0:
            value = float('-inf')
            for child in children:
                child_val = self.alphabeta(
                    child, child.state,
                    depth - 1, alpha, beta,
                    1, initial_time, seen_nodes
                )
                value = max(value, child_val)     
                alpha = max(alpha, value)           
                if alpha >= beta:
                    break

        # MIN PLAYER
        else:
            value = float('inf')
            for child in children:
                child_val = self.alphabeta(
                    child, child.state,
                    depth - 1, alpha, beta,
                    0, initial_time, seen_nodes
                )
                value = min(value, child_val)
                beta = min(beta, value)
                if beta <= alpha:
                    break

        seen_nodes[k] = [depth, value]
        return value


    # -----------------------------------------------------------
    # DEPTH SEARCH
    def depth_search(self, node, depth, initial_time, seen_nodes):
        alpha = float('-inf')
        beta = float('inf')
        children = node.compute_and_get_children()

        scores = []
        for child in children:
            score = self.alphabeta(
                child, child.state,
                depth, alpha, beta,
                1, initial_time, seen_nodes
            )
            scores.append(score)

        best_index = scores.index(max(scores))
        return children[best_index].move


    # -----------------------------------------------------------
    # MAIN SEARCH FUNCTION
    def search_best_next_move(self, initial_tree_node):

        initial_time = time.time()
        depth = 0
        timeout = False
        seen_nodes = dict()
        best_move = 0

        while not timeout and depth <= MAX_DEPTH:
            try:
                move = self.depth_search(
                    initial_tree_node, depth,
                    initial_time, seen_nodes
                )
                best_move = move
                depth += 1

            except TimeoutError:
                timeout = True

        return ACTION_TO_STR[best_move]




