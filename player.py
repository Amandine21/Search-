import random
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR




MAX_DEPTH = 3

class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the alpha-beta next move search.
        """
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            node = Node(message=msg, player=0)

            # Use alpha-beta to select best move
            best_move = self.search_best_next_move_alphabeta(
                root=node,
                evaluate=self.evaluate_node,
                max_depth=MAX_DEPTH,
                time_limit=0.55
            )

            self.sender({"action": best_move, "search_time": None})

    # ----------------- EVALUATION FUNCTION -----------------
    def evaluate_node(self, node):
        state = node.state
        fish_positions = state.get_fish_positions()
        hooks = state.get_hook_positions()

        def torus_manhattan(a, b, width=20):
            dx = abs(a[0] - b[0])
            dx = min(dx, width - dx)
            dy = abs(a[1] - b[1])
            return dx + dy

        best_dist_max = float("inf")
        best_dist_min = float("inf")

        for pos in fish_positions.values():
            d0 = torus_manhattan(hooks[0], pos)  # MAX (green)
            d1 = torus_manhattan(hooks[1], pos)  # MIN (red)

            best_dist_max = min(best_dist_max, d0)
            best_dist_min = min(best_dist_min, d1)

        # Simple evaluation: distance from MAX hook to nearest fish
        return best_dist_min - best_dist_max

    # ----------------- ALPHA-BETA PRUNING -----------------
    def alphabeta(self, node, depth, alpha, beta, maximizing_player, evaluate, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            raise TimeoutError

        children = node.compute_and_get_children()
        if depth == 0 or not children:
            return evaluate(node)

        if maximizing_player:
            value = float("-inf")
            for child in children:
                score = self.alphabeta(
                    child, depth-1, alpha, beta, False,
                    evaluate, start_time, time_limit
                )
                value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # beta prune
            return value
        else:
            value = float("inf")
            for child in children:
                score = self.alphabeta(
                    child, depth-1, alpha, beta, True,
                    evaluate, start_time, time_limit
                )
                value = min(value, score)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # alpha prune
            return value

    # ----------------- BEST MOVE SELECTION -----------------
    def search_best_next_move_alphabeta(self, root, evaluate, max_depth=MAX_DEPTH, time_limit=0.55):
        root_children = root.compute_and_get_children()
        if not root_children:
            return ACTION_TO_STR[0]

        start_time = time.time()
        best_value = float("-inf")
        best_moves = []

        try:
            for child in root_children:
                score = self.alphabeta(
                    child,
                    max_depth-1,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    maximizing_player=False,
                    evaluate=evaluate,
                    start_time=start_time,
                    time_limit=time_limit
                )

                if score > best_value:
                    best_value = score
                    best_moves = [child.move]
                elif score == best_value:
                    best_moves.append(child.move)

        except TimeoutError:
            # Time ran out; just use the best moves found so far
            pass

        # Ensure we always return a valid move
        if not best_moves:
            return ACTION_TO_STR[0]

        return ACTION_TO_STR[random.choice(best_moves)]
