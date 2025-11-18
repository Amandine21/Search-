'''
CODE THAT WORKS (KATTIS - 21 POINTS)
import time
import random
from collections import defaultdict

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


# === TUNABLE PARAMETERS ===
TIME_LIMIT = 0.055            
MAX_DEPTH = 15              
KILLER_SLOTS = 2            
QUIESCENCE_DEPTH = 2          

class TimeUp(Exception):
    pass


class TTEntry:
    def __init__(self, value, depth, flag, best_move):
        self.value = value
        self.depth = depth     # depth of stored result
        self.flag = flag        # EXACT, LOWER, UPPER
        self.best_move = best_move


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super().__init__()
        self.tt = {}                   # Transposition table
        self.killers = defaultdict(lambda: [])
        self.pv_table = {}

    # ===================================================================
    #                GAME LOOP
    # ===================================================================
    def player_loop(self):
        _ = self.receiver()  # first message required by engine

        while True:
            msg = self.receiver()
            root = Node(message=msg, player=0)
            move = self.search_best_next_move(root)
            self.sender({"action": move, "search_time": None})

    # ===================================================================
    #                HASH KEY
    # ===================================================================
    def hash_key(self, state):
        # Hook positions
        hooks = state.get_hook_positions()
        key = f"H{hooks[0]}-{hooks[1]}"

        # Sorted fish positions + scores
        fish_pos = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        for fid in sorted(fish_pos.keys()):
            key += f"|F{fid}:{fish_pos[fid]}:{fish_scores[fid]}"

        # Include caught fish & current player to avoid collisions
        c0, c1 = state.get_caught()
        key += f"|C{c0},{c1}|P{state.get_player()}"

        return key

    # ===================================================================
    #                TORUS DISTANCE
    # ===================================================================
    def torus_dist(self, a, b, size=20):
        dx = abs(a[0] - b[0])
        dx = min(dx, size - dx)
        dy = abs(a[1] - b[1])
        return dx + dy

    # ===================================================================
    #                HEURISTIC (TUNED FOR 20/25 KATTIS)
    # ===================================================================
    def evaluate_state(self, state):
        """
        Improved heuristic tuned for Kattis:
        - Encourages immediate captures
        - Prefers high-value fish even if slightly farther
        - Penalizes letting opponent get closer to valuable fish
        - Uses shallower distance exponent to prevent 'snap chasing'
        """

        hooks = state.get_hook_positions()
        fish_pos = state.get_fish_positions()
        fish_scores = state.get_fish_scores()

        # Player score difference
        g_score, r_score = state.get_player_scores()
        score_diff = g_score - r_score

        # Weighted score difference
        value = 4.2 * score_diff

        g_hook = hooks[0]
        r_hook = hooks[1]

        EXP = 1.8   # shallower exponent gives smoother gradients

        for fid, pos in fish_pos.items():
            fish_val = float(fish_scores[fid])

            # torus distances
            d_g = self.torus_dist(g_hook, pos) + 1.0
            d_r = self.torus_dist(r_hook, pos) + 1.0

            # ---- Immediate catch bonus ----
            if pos == g_hook and fish_val > 0:
                value += 900      # ensures guaranteed grab
            if pos == r_hook and fish_val > 0:
                value -= 900      # deny opponent grabs

            # ---- Attraction to fish ----
            my_att = fish_val / (d_g ** EXP)
            opp_att = fish_val / (d_r ** EXP)

            # ---- Relative positional advantage ----
            # reward being closer than opponent to valuable fish
            dist_adv = (d_r - d_g) * 0.35 * fish_val

            # ---- Vertical alignment heuristic ----
            # fishes line up under hooks frequently in Kattis maps
            vertical_bonus = 0
            if abs(g_hook[0] - pos[0]) <= 1:
                vertical_bonus += fish_val * 0.25
            if abs(r_hook[0] - pos[0]) <= 1:
                vertical_bonus -= fish_val * 0.25

            value += my_att - opp_att + dist_adv + vertical_bonus

        return value


    # ===================================================================
    #               QUIESCENCE SEARCH
    # ===================================================================
    def quiescence(self, node, alpha, beta, start_t, depth=QUIESCENCE_DEPTH):
        if time.perf_counter() - start_t > TIME_LIMIT:
            raise TimeUp

        stand = self.evaluate_state(node.state)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        if depth <= 0:
            return stand

        children = node.compute_and_get_children()
        capture_children = []
        for child in children:
            st = child.state
            hooks = st.get_hook_positions()
            fishes = st.get_fish_positions()

            # if any fish is caught in this child → quiescence-extend
            for p in fishes.values():
                if p == hooks[0] or p == hooks[1]:
                    capture_children.append(child)
                    break

        if not capture_children:
            return stand

        # Order by static eval
        capture_children.sort(key=lambda c: self.evaluate_state(c.state), reverse=True)

        for ch in capture_children:
            val = -self.quiescence(ch, -beta, -alpha, start_t, depth - 1)
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val

        return alpha

    # ===================================================================
    #               ALPHA-BETA WITH TT + KILLER + PV
    # ===================================================================
    def alphabeta(self, node, depth, alpha, beta, player, start_t):
        if time.perf_counter() - start_t > TIME_LIMIT:
            raise TimeUp

        state = node.state
        key = self.hash_key(state)

        # ---- Transposition Table Lookup ----
        if key in self.tt:
            entry = self.tt[key]
            if entry.depth >= depth:
                # EXACT value
                if entry.flag == 'EXACT':
                    return entry.value
                # Lower bound
                if entry.flag == 'LOWER':
                    alpha = max(alpha, entry.value)
                # Upper bound
                if entry.flag == 'UPPER':
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value

        # ---- Leaf or no children ----
        children = node.compute_and_get_children()
        if depth == 0 or not children:
            return self.quiescence(node, alpha, beta, start_t)

        # ==== MOVE ORDERING ====
        moves_ordered = []
        pv_move = self.pv_table.get(key, None)
        killers = self.killers[depth]

        for ch in children:
            score = 0.0

            if ch.move == pv_move:
                score += 1e6
            if ch.move in killers:
                score += 5e5

            score += self.evaluate_state(ch.state)

            moves_ordered.append((score, ch))

        moves_ordered.sort(key=lambda t: t[0], reverse=True)

        best_value = float('-inf') if player == 0 else float('inf')
        best_move = None
        flag = 'UPPER' if player == 1 else 'LOWER'

        # ==== MAX PLAYER ====
        if player == 0:
            a = alpha
            for _, ch in moves_ordered:
                val = self.alphabeta(ch, depth - 1, a, beta, 1, start_t)
                if val > best_value:
                    best_value = val
                    best_move = ch.move
                a = max(a, best_value)
                if a >= beta:
                    # killer moves
                    if best_move is not None:
                        lst = self.killers[depth]
                        if best_move not in lst:
                            lst.insert(0, best_move)
                            if len(lst) > KILLER_SLOTS:
                                lst.pop()
                    flag = 'LOWER'
                    break

            alpha = max(alpha, best_value)

        # ==== MIN PLAYER ====
        else:
            b = beta
            for _, ch in moves_ordered:
                val = self.alphabeta(ch, depth - 1, alpha, b, 0, start_t)
                if val < best_value:
                    best_value = val
                    best_move = ch.move
                b = min(b, best_value)
                if b <= alpha:
                    if best_move is not None:
                        lst = self.killers[depth]
                        if best_move not in lst:
                            lst.insert(0, best_move)
                            if len(lst) > KILLER_SLOTS:
                                lst.pop()
                    flag = 'UPPER'
                    break

            beta = min(beta, best_value)

        # ==== TT STORE ====
        if alpha < best_value < beta:
            flag = 'EXACT'

        self.tt[key] = TTEntry(best_value, depth, flag, best_move)
        if best_move is not None:
            self.pv_table[key] = best_move

        return best_value

    # ===================================================================
    #               DEPTH SEARCH
    # ===================================================================
    def depth_search(self, root, depth, start_t):
        alpha = float('-inf')
        beta = float('inf')

        children = root.compute_and_get_children()
        if not children:
            return 4  # fallback STAY

        ordered = []
        for ch in children:
            score = self.evaluate_state(ch.state)
            ordered.append((score, ch))

        ordered.sort(key=lambda t: t[0], reverse=True)

        best_value = float('-inf')
        best_move = ordered[0][1].move

        for _, ch in ordered:
            val = self.alphabeta(ch, depth - 1, alpha, beta, 1, start_t)
            if val > best_value:
                best_value = val
                best_move = ch.move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_move

    # ===================================================================
    #               ITERATIVE DEEPENING
    # ===================================================================
    def search_best_next_move(self, root):
        start_t = time.perf_counter()

        best_move = 4  # fallback STAY
        self.killers.clear()
        self.pv_table.clear()

        for depth in range(1, MAX_DEPTH + 1):
            try:
                move = self.depth_search(root, depth, start_t)
                best_move = move
                if time.perf_counter() - start_t > TIME_LIMIT:
                    break
            except TimeUp:
                break
            except Exception:
                break

        return ACTION_TO_STR[best_move]
'''
# "WINDOWS" CODE
import time
from collections import defaultdict

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


# === TUNABLE PARAMETERS ===
TIME_LIMIT = 0.055
MAX_DEPTH = 15
KILLER_SLOTS = 2
QUIESCENCE_DEPTH = 2


class TimeUp(Exception):
    pass


class TTEntry:
    def __init__(self, value, depth, flag, best_move):
        self.value = value
        self.depth = depth
        self.flag = flag    # EXACT, LOWER, UPPER
        self.best_move = best_move


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super().__init__()
        self.tt = {}
        self.killers = defaultdict(list)  # picklable
        self.pv_table = {}

    # ===================================================================
    #                GAME LOOP
    # ===================================================================
    def player_loop(self):
        _ = self.receiver()  # first message required by engine

        while True:
            msg = self.receiver()
            root = Node(message=msg, player=0)
            move_str = self.search_best_next_move(root)
            self.sender({"action": move_str, "search_time": None})

    # ===================================================================
    #                HASH KEY
    # ===================================================================
    def hash_key(self, state):
        hooks = state.get_hook_positions()
        key = f"H{hooks[0]}-{hooks[1]}"

        fish_pos = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        for fid in sorted(fish_pos.keys()):
            key += f"|F{fid}:{fish_pos[fid]}:{fish_scores[fid]}"

        c0, c1 = state.get_caught()
        key += f"|C{c0},{c1}|P{state.get_player()}"

        return key

    # ===================================================================
    #                TORUS DISTANCE
    # ===================================================================
    def torus_dist(self, a, b, size=20):
        dx = abs(a[0] - b[0])
        dx = min(dx, size - dx)
        dy = abs(a[1] - b[1])
        return dx + dy

    # ===================================================================
    #                HEURISTIC
    # ===================================================================
    def evaluate_state(self, state):
        hooks = state.get_hook_positions()
        fish_pos = state.get_fish_positions()
        fish_scores = state.get_fish_scores()

        g_score, r_score = state.get_player_scores()
        score_diff = g_score - r_score
        value = 4.2 * score_diff

        g_hook = hooks[0]
        r_hook = hooks[1]

        EXP = 1.8

        for fid, pos in fish_pos.items():
            fish_val = float(fish_scores[fid])

            d_g = self.torus_dist(g_hook, pos) + 1.0
            d_r = self.torus_dist(r_hook, pos) + 1.0

            if pos == g_hook and fish_val > 0:
                value += 900
            if pos == r_hook and fish_val > 0:
                value -= 900

            my_att = fish_val / (d_g ** EXP)
            opp_att = fish_val / (d_r ** EXP)

            dist_adv = (d_r - d_g) * 0.35 * fish_val

            vertical_bonus = 0
            if abs(g_hook[0] - pos[0]) <= 1:
                vertical_bonus += fish_val * 0.25
            if abs(r_hook[0] - pos[0]) <= 1:
                vertical_bonus -= fish_val * 0.25

            value += my_att - opp_att + dist_adv + vertical_bonus

        return value

    # ===================================================================
    #               QUIESCENCE SEARCH
    # ===================================================================
    def quiescence(self, node, alpha, beta, start_t, depth=QUIESCENCE_DEPTH):
        if time.perf_counter() - start_t > TIME_LIMIT:
            raise TimeUp

        stand = self.evaluate_state(node.state)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

        if depth <= 0:
            return stand

        children = node.compute_and_get_children()
        capture_children = []
        for child in children:
            st = child.state
            hooks = st.get_hook_positions()
            fishes = st.get_fish_positions()
            for p in fishes.values():
                if p == hooks[0] or p == hooks[1]:
                    capture_children.append(child)
                    break

        if not capture_children:
            return stand

        capture_children.sort(
            key=lambda c: self.evaluate_state(c.state),
            reverse=True
        )

        for ch in capture_children:
            val = -self.quiescence(ch, -beta, -alpha, start_t, depth - 1)
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val

        return alpha

    # ===================================================================
    #               ALPHA-BETA WITH TT + KILLER + PV
    # ===================================================================
    def alphabeta(self, node, depth, alpha, beta, player, start_t):
        if time.perf_counter() - start_t > TIME_LIMIT:
            raise TimeUp

        state = node.state
        key = self.hash_key(state)

        if key in self.tt:
            entry = self.tt[key]
            if entry.depth >= depth:
                if entry.flag == 'EXACT':
                    return entry.value
                if entry.flag == 'LOWER':
                    alpha = max(alpha, entry.value)
                if entry.flag == 'UPPER':
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value

        children = node.compute_and_get_children()
        if depth == 0 or not children:
            return self.quiescence(node, alpha, beta, start_t)

        moves_ordered = []
        pv_move = self.pv_table.get(key, None)
        killers = self.killers[depth]

        for ch in children:
            score = 0.0
            if ch.move == pv_move:
                score += 1e6
            if ch.move in killers:
                score += 5e5
            score += self.evaluate_state(ch.state)
            moves_ordered.append((score, ch))

        moves_ordered.sort(key=lambda t: t[0], reverse=True)

        best_value = float('-inf') if player == 0 else float('inf')
        best_move = None
        flag = 'UPPER' if player == 1 else 'LOWER'

        if player == 0:
            a = alpha
            for _, ch in moves_ordered:
                val = self.alphabeta(ch, depth - 1, a, beta, 1, start_t)
                if val > best_value:
                    best_value = val
                    best_move = ch.move
                a = max(a, best_value)
                if a >= beta:
                    if best_move is not None:
                        lst = self.killers[depth]
                        if best_move not in lst:
                            lst.insert(0, best_move)
                            if len(lst) > KILLER_SLOTS:
                                lst.pop()
                    flag = 'LOWER'
                    break
            alpha = max(alpha, best_value)

        else:
            b = beta
            for _, ch in moves_ordered:
                val = self.alphabeta(ch, depth - 1, alpha, b, 0, start_t)
                if val < best_value:
                    best_value = val
                    best_move = ch.move
                b = min(b, best_value)
                if b <= alpha:
                    if best_move is not None:
                        lst = self.killers[depth]
                        if best_move not in lst:
                            lst.insert(0, best_move)
                            if len(lst) > KILLER_SLOTS:
                                lst.pop()
                    flag = 'UPPER'
                    break
            beta = min(beta, best_value)

        if alpha < best_value < beta:
            flag = 'EXACT'

        self.tt[key] = TTEntry(best_value, depth, flag, best_move)
        if best_move is not None:
            self.pv_table[key] = best_move

        return best_value

    # ===================================================================
    #               DEPTH SEARCH
    # ===================================================================
    def depth_search(self, root, depth, start_t):
        alpha = float('-inf')
        beta = float('inf')

        children = root.compute_and_get_children()
        if not children:
            return 4

        ordered = []
        for ch in children:
            ordered.append((self.evaluate_state(ch.state), ch))

        ordered.sort(key=lambda t: t[0], reverse=True)

        best_value = float('-inf')
        best_move = ordered[0][1].move

        for _, ch in ordered:
            val = self.alphabeta(ch, depth - 1, alpha, beta, 1, start_t)
            if val > best_value:
                best_value = val
                best_move = ch.move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        return best_move

    # ===================================================================
    #               ITERATIVE DEEPENING
    # ===================================================================
    def search_best_next_move(self, root):
        start_t = time.perf_counter()

        best_move = 4
        self.killers.clear()
        self.pv_table.clear()

        for depth in range(1, MAX_DEPTH + 1):
            try:
                move = self.depth_search(root, depth, start_t)
                best_move = move
                if time.perf_counter() - start_t > TIME_LIMIT:
                    break
            except TimeUp:
                break
            except Exception:
                break

        return ACTION_TO_STR[best_move]
