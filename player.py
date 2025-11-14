import random
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


# Creates a unique identifier for a game statee so it can be stored and reused in a memorization table
# The game state is a complete snapshot of the fishing game at a SPECIFIC MOMENT which includes the position of fishing hooks, position of all the fish, the score of  both players, an the current direction and movement of the fish
# Everything the engine needs to know to fully continue the game from that exact moment

def compute_hash(node):
    return node.state.serialize() 


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax to find the best possible next move for player 0 (green boat).
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # ------------ PARAMETERS ------------
        # How many plies (half-moves) to look ahead from the ROOT ACTION.
        # We call minimax on the children of the root with depth = MAX_DEPTH - 1.
        MAX_DEPTH = 6

        # ---------- HELPER: EVALUATION ----------
        def evaluate(node):
            """
            Heuristic evaluation function γ(state):
            simply use the score difference from MAX's perspective.
            This is basically the heuristic from task 2.1
            """
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

                if d0 < best_dist_max:
                    best_dist_max = d0
                if d1 < best_dist_min:
                    best_dist_min = d1

            return best_dist_min - best_dist_max

        # ---------- HELPER: MINIMAX ----------
        def minimax(node, depth, maximizing_player):
            """
            Recursive minimax implementation with fixed depth.
            :param node: current Node
            :param depth: remaining depth to search
            :param maximizing_player: True if it's MAX's turn, False if MIN's
            :return: heuristic value
            """

            # Generate children for this node (if not already done)
            children = node.compute_and_get_children()

            # Terminal / depth limit: evaluate
            if depth == 0 or len(children) == 0:
                return evaluate(node)

            if maximizing_player:
                best_val = float("-inf")
                for child in children:
                    val = minimax(child, depth - 1, False)
                    if val > best_val:
                        best_val = val
                return best_val
            else:
                best_val = float("inf")
                for child in children:
                    val = minimax(child, depth - 1, True)
                    if val < best_val:
                        best_val = val
                return best_val

        # ---------- ROOT DECISION ----------
        # Expand root children: these correspond to our 5 possible actions (0..4)
        root_children = initial_tree_node.compute_and_get_children()

        # If for some reason we have no children (e.g., game over), just stay.
        if not root_children:
            return ACTION_TO_STR[0]  # "stay"

        best_value = float("-inf")
        best_moves = []

        for child in root_children:
            # After our move, it becomes opponent's (player 1) turn -> minimizing.
            value = minimax(child, MAX_DEPTH - 1, maximizing_player=False)

            if value > best_value:
                best_value = value
                best_moves = [child.move]
            elif value == best_value:
                best_moves.append(child.move)

        # Break ties randomly to avoid being predictable
        chosen_action_int = random.choice(best_moves)
        return ACTION_TO_STR[chosen_action_int]
    

       # ---------- ALPHA BETA PRUNNING ----------
    def alphabeta(self, node, depth, alpha, beta, maximizing_player, evaluate, start_time, time_limit, visited):
        
        # Time out Error Checking
        if time.time() - start_time >= 0.55:
            raise TimeoutError
        
        # Memorization Table (If already visited  this state with equal)
        key = compute_hash(node)
        
        # Check if we visited the node. If we visited the node and the depth is greater that means we found a state that is more accurate
        # Return the heurstic score for that node!
        if key in visited and visited[key][0] >= depth:
            return visited[key][1]
        
        # Get the children of the current node
        children = node.compute_and_get_children()

        # Check the base case of the search and compute the heuristic value of the node
        if(depth == 0 or len(children) == 0):
            value = evaluate(node)
            visited[key] = (depth, value)
            return value
        

        # ---------------- MAXIMIZING PLAYER ----------------
        # Define the logic for the maximing player turns (levels in the tree)
        if maximizing_player:
            value = float("-inf")

            # Calculate the heurstic score for every child node to find how good the game state is for the maximizing player
            for child in children:
                    score = self.alphabeta(
                        child,
                        depth - 1,
                        alpha,
                        beta,
                        False,            
                        evaluate,
                        start_time,
                        time_limit,
                        visited
                    )   

                    value = max(value, score)

                    # Update the alpha value, which is the best value that maximizing player can gaurantee in the current game state
                    alpha = max(alpha, value)

                    # Comparision for Prunning
                    if alpha >= beta:
                        break
            
        # ---------------- MINIMIZING PLAYER ----------------
        else:
            value = float("inf")

            for child in children:
                score = self.alphabeta(
                    child,
                    depth - 1,
                    alpha,
                    beta,
                    True,             
                    evaluate,
                    start_time,
                    time_limit,
                    visited
                )

                # Comptue the herustic values
                value = min(value, score)

                # Compute the beta value by finding the min value
                beta = min(beta, value)

                # Comparision for prunning!
                if(beta <= alpha):
                    break

        # Store in the memo table
        visited[key] = (depth, value)
        return value
    




    def search_best_next_move_alphabeta(self, root, evaluate, max_depth=5, time_limit=0.055):
        """
        Root-level decision using alpha-beta pruning.
        This function is SEPARATE so it does not modify your existing search_best_next_move().
        """

        root_children = root.compute_and_get_children()
        if not root_children:
            return ACTION_TO_STR[0]

        start_time = time.time()
        visited = {}

        best_value = float("-inf")
        best_moves = []

        for child in root_children:
            try:
                score = self.alphabeta(
                    child,
                    max_depth - 1,          # one ply consumed
                    alpha=float("-inf"),
                    beta=float("inf"),
                    maximizing_player=False, # opponent turn
                    evaluate=evaluate,
                    start_time=start_time,
                    time_limit=time_limit,
                    visited=visited
                )
            except TimeoutError:
                break

            if score > best_value:
                best_value = score
                best_moves = [child.move]
            elif score == best_value:
                best_moves.append(child.move)

        return ACTION_TO_STR[random.choice(best_moves)]
