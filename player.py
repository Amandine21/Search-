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
        fish_scores = state.get_fish_scores()
        caught_fish = state.get_caught()

        def torus_manhattan(a, b, width=20):
            dx = abs(a[0] - b[0])
            dx = min(dx, width - dx)
            dy = abs(a[1] - b[1])
            return dx + dy

        max_score = 0
        min_score = 0

        for fish_id, pos in fish_positions.items():
            fish_value = fish_scores[fish_id]

            # Distance to hooks
            d_max = torus_manhattan(hooks[0], pos) + 1
            d_min = torus_manhattan(hooks[1], pos) + 1

            # If hook is on the fish, it counts as caught
            if hooks[0] == pos:
                max_score += fish_value  # full value
            else:
                max_score += fish_value / d_max  # partial value if approaching

            if hooks[1] == pos:
                min_score += fish_value
            else:
                min_score += fish_value / d_min

        return max_score - min_score


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
