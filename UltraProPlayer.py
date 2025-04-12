# from LegendPlayer import LegendHexPlayer
from Player import HexPlayer
from Board import HexBoard
import random
import time
import heapq

class UltraProHexPlayer(HexPlayer):
    zobrist_table = None

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.transposition_table = {}
        self.neighbor_cache = {}
        self.move_history = []       # [(board_key, move)]
        self.move_stats = {}         # {(board_key, move): score}
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
        self.max_depth = 3

    def init_zobrist_table(self, size):
        if UltraProHexPlayer.zobrist_table is None or len(UltraProHexPlayer.zobrist_table) != size:
            # Inicializar tabla [size][size][3] para 0,1,2
            UltraProHexPlayer.zobrist_table = [
                [ [random.getrandbits(64) for _ in range(3)] for _ in range(size) ]
                for _ in range(size)
            ]

    def play(self, board: HexBoard) -> tuple:
        # Inicializar Zobrist si es primera jugada
        self.init_zobrist_table(board.size)

        # Limpiar transposición si crece mucho
        if len(self.transposition_table) > 200000:
            self.transposition_table.clear()

        possible_moves = board.get_possible_moves()
        total_cells = board.size * board.size

        # Apertura: usar MCTS si tablero muy vacío
        if len(possible_moves) > total_cells * 0.8:
            return self._mcts_select_move(board, simulations=200, top_k=10)

        # Primera jugada: centro
        if len(possible_moves) == total_cells:
            return (board.size // 2, board.size // 2)

        # Victoria o bloqueo inmediato
        for move in possible_moves:
            b2 = board.clone()
            b2.place_piece(move[0], move[1], self.player_id)
            if b2.check_connection(self.player_id):
                return move
        for move in possible_moves:
            b2 = board.clone()
            b2.place_piece(move[0], move[1], self.opponent_id)
            if b2.check_connection(self.opponent_id):
                return move

        # Ajuste profundidad según fase
        moves_played = total_cells - len(possible_moves)
        ratio = moves_played / total_cells
        if ratio < 0.2:
            self.max_depth = 2
        elif ratio < 0.6:
            self.max_depth = 3
        else:
            self.max_depth = 4

        # Ordenar y limitar movimientos
        move_limit = min(12, len(possible_moves))
        ordered = sorted(
            possible_moves,
            key=lambda m: self._move_priority(board, m),
            reverse=True
        )[:move_limit]

        best_move, best_val = ordered[0], float('-inf')
        alpha, beta = float('-inf'), float('inf')

        for move in ordered:
            b2 = board.clone()
            b2.place_piece(move[0], move[1], self.player_id)
            val = -self.alpha_beta(b2, self.max_depth-1, -beta, -alpha, False)
            if val > best_val:
                best_val, best_move = val, move
                alpha = max(alpha, val)

        # Guardar en historial
        key = self._normalized_board_key(board)
        self.move_history.append((key, best_move))
        if len(self.move_history) > 200:
            self.move_history.pop(0)

        return best_move

    def alpha_beta(self, board, depth, alpha, beta, maximizing):
        key = self._normalized_board_key(board)
        if key in self.transposition_table:
            return self.transposition_table[key]

        if board.check_connection(self.player_id):
            return 10000 + depth
        if board.check_connection(self.opponent_id):
            return -10000 - depth
        if depth == 0:
            score = self._fast_evaluation(board)
            self.transposition_table[key] = score
            return score

        moves = board.get_possible_moves()
        if not moves:
            return 0

        if maximizing:
            value = float('-inf')
            ordered = sorted(moves, key=lambda m: self._move_priority(board, m), reverse=True)[:8]
            for m in ordered:
                b2 = board.clone(); b2.place_piece(m[0], m[1], self.player_id)
                value = max(value, self.alpha_beta(b2, depth-1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float('inf')
            ordered = sorted(moves, key=lambda m: self._move_priority_opponent(board, m), reverse=True)[:8]
            for m in ordered:
                b2 = board.clone(); b2.place_piece(m[0], m[1], self.opponent_id)
                value = min(value, self.alpha_beta(b2, depth-1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break

        self.transposition_table[key] = value
        return value
    
    def _move_priority(self, board, move):
        """Heurística mejorada para priorizar movimientos para nosotros"""
        i, j = move
        size = board.size
        priority = 0
        
        # Valor base por posición estratégica
        center = size // 2
        
        # Para jugador horizontal (1), priorizar movimientos que avanzan horizontalmente
        if self.player_id == 1:
            # Mayor prioridad a movimientos en la columna central
            priority += (1 - abs(i - center) / center) * 5
            # Bonus por avance horizontal
            priority += j * 2 / size
        # Para jugador vertical (2), priorizar movimientos que avanzan verticalmente
        else:
            # Mayor prioridad a movimientos en la fila central
            priority += (1 - abs(j - center) / center) * 5
            # Bonus por avance vertical
            priority += i * 2 / size
        
        # Bonus por proximidad a fichas propias
        own_neighbors = 0
        for neighbor in self.get_neighbors(i, j, size):
            ni, nj = neighbor
            if board.board[ni][nj] == self.player_id:
                own_neighbors += 1
                # Mayor valor a conexiones que mantienen continuidad
                if (self.player_id == 1 and ni == i) or (self.player_id == 2 and nj == j):
                    priority += 1
        
        priority += own_neighbors * 3
        
        # Bonus por crear puentes
        for d1, d2 in [(0, 2), (2, 0), (2, -2), (-2, 2), (-2, 0), (0, -2)]:
            ni, nj = i + d1, j + d2
            if 0 <= ni < size and 0 <= nj < size and board.board[ni][nj] == self.player_id:
                priority += 4  # Alto valor a completar puentes
        
        # Penalizar movimientos que benefician al oponente
        opponent_neighbors = 0
        for neighbor in self.get_neighbors(i, j, size):
            ni, nj = neighbor
            if board.board[ni][nj] == self.opponent_id:
                opponent_neighbors += 1
        
        priority -= opponent_neighbors * 0.5  # Ligera penalización
        
        return priority

    def _move_priority_opponent(self, board, move):
        """Evalúa movimientos desde perspectiva del oponente"""
        i, j = move
        size = board.size
        priority = 0
        
        # Evaluar posición estratégica para el oponente
        center = size // 2
        
        # Aplicar lógica inversa a la del jugador principal
        if self.opponent_id == 1:
            priority += (1 - abs(i - center) / center) * 5
            priority += j * 2 / size
        else:
            priority += (1 - abs(j - center) / center) * 5
            priority += i * 2 / size
        
        # Priorizar cercanía a otras fichas del oponente
        for neighbor in self.get_neighbors(i, j, size):
            ni, nj = neighbor
            if board.board[ni][nj] == self.opponent_id:
                priority += 3
                if (self.opponent_id == 1 and ni == i) or (self.opponent_id == 2 and nj == j):
                    priority += 1
        
        return priority
    
    def _calculate_min_distance(self, board, player_id):
        """Dijkstra optimizado para calcular distancia mínima entre bordes"""
        size = board.size
        
        # Definir nodos de inicio y destino según el jugador
        if player_id == 1:  # Horizontal (izquierda a derecha)
            starts = [(i, 0) for i in range(size)]
            ends = set((i, size - 1) for i in range(size))
        else:  # Vertical (arriba a abajo)
            starts = [(0, j) for j in range(size)]
            ends = set((size - 1, j) for j in range(size))
        
        # Inicializar variables para Dijkstra
        heap = []
        distances = {}
        
        # Agregar nodos iniciales
        for start in starts:
            i, j = start
            if board.board[i][j] == player_id:
                heapq.heappush(heap, (0, start))
                distances[start] = 0
            elif board.board[i][j] == 0:  # Casilla vacía
                heapq.heappush(heap, (1, start))
                distances[start] = 1
        
        # Si no hay puntos de inicio válidos, retornar distancia máxima
        if not heap:
            return size * 2
        
        # Algoritmo de Dijkstra
        while heap:
            dist, current = heapq.heappop(heap)
            
            # Si llegamos a un punto final, terminamos
            if current in ends:
                return dist
                
            # Evitar procesamiento repetido
            if distances.get(current, float('inf')) < dist:
                continue
                
            # Explorar vecinos
            for neighbor in self.get_neighbors(current[0], current[1], size):
                ni, nj = neighbor
                # Costo: 0 si ya tenemos ficha, 1 si está vacío, infinito si es del oponente
                if board.board[ni][nj] == player_id:
                    cost = 0
                elif board.board[ni][nj] == 0:
                    cost = 1
                else:
                    continue  # No podemos pasar por casillas del oponente
                    
                new_dist = dist + cost
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        
        # Si no encontramos camino, retornar distancia máxima
        return size * 2
    
    def _calculate_territory_control(self, board, player_id):
        """Evalúa el control de territorio basado en influencia de casillas"""
        size = board.size
        control = 0
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                cell_value = board.board[i][j]
                if cell_value == player_id:
                    # Valor base por cada ficha propia
                    control += 1
                    
                    # Bonus por posición estratégica
                    if player_id == 1:  # Jugador horizontal
                        position_bonus = (1 - abs(i - center) / center) * 0.5  # Bonus por estar cerca del centro vertical
                    else:  # Jugador vertical
                        position_bonus = (1 - abs(j - center) / center) * 0.5  # Bonus por estar cerca del centro horizontal
                    
                    control += position_bonus
                    
                    # Bonus por conexión con otras fichas propias
                    connectivity = 0
                    for neighbor in self.get_neighbors(i, j, size):
                        ni, nj = neighbor
                        if board.board[ni][nj] == player_id:
                            connectivity += 1
                    
                    # Mayor valor a las fichas conectadas
                    control += connectivity * 0.2
                    
        return control
    
    def _pattern_evaluation(self, board):
        """Evalúa patrones conocidos que son favorables o desfavorables"""
        size = board.size
        score = 0
        
        # Buscar patrones de puente (dos fichas separadas por un espacio)
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == self.player_id:
                    # Verificar puentes en las 6 direcciones
                    for d1, d2 in [(0, 2), (2, 0), (2, -2), (-2, 2), (-2, 0), (0, -2)]:
                        ni, nj = i + d1, j + d2
                        if 0 <= ni < size and 0 <= nj < size and board.board[ni][nj] == self.player_id:
                            # Verificar si el espacio intermedio está vacío
                            mi, mj = i + d1//2, j + d2//2
                            if 0 <= mi < size and 0 <= mj < size and board.board[mi][mj] == 0:
                                score += 2  # Puente es una estructura fuerte
        
        # Penalizar fichas aisladas
        for i in range(size):
            for j in range(size):
                if board.board[i][j] == self.player_id:
                    isolated = True
                    for neighbor in self.get_neighbors(i, j, size):
                        ni, nj = neighbor
                        if board.board[ni][nj] == self.player_id:
                            isolated = False
                            break
                    if isolated:
                        score -= 1  # Penalizar ficha aislada
        
        return score


    def _board_key(self, board: HexBoard) -> int:
        # Zobrist hashing sin simetría
        h = 0
        for i in range(board.size):
            for j in range(board.size):
                p = board.board[i][j]
                if p:
                    h ^= UltraProHexPlayer.zobrist_table[i][j][p]
        return h

    def generate_symmetries(self, board: HexBoard):
        """Genera variaciones del tablero: identidad, flip horizontal, flip vertical, ambas."""
        syms = []
        orig = board.board
        size = board.size
        # Identidad
        syms.append(orig)
        # Flip horizontal (filas invertidas)
        syms.append(list(reversed(orig)))
        # Flip vertical (columnas invertidas)
        syms.append([list(reversed(row)) for row in orig])
        # Flip ambas
        syms.append([list(reversed(row)) for row in reversed(orig)])
        boards = []
        for mat in syms:
            b = board.clone()
            b.board = [row[:] for row in mat]
            boards.append(b)
        return boards

    def _normalized_board_key(self, board: HexBoard) -> int:
        """Retorna el menor hash Zobrist entre todas las simetrías"""
        keys = [self._board_key(b) for b in self.generate_symmetries(board)]
        return min(keys)

    def _fast_evaluation(self, board: HexBoard) -> float:
        # Similar a LegendHexPlayer: distancia + control + patrones
        pd = self._calculate_min_distance(board, self.player_id)
        od = self._calculate_min_distance(board, self.opponent_id)
        pc = self._calculate_territory_control(board, self.player_id)
        oc = self._calculate_territory_control(board, self.opponent_id)
        dist_score = (od - pd) * 60
        ctrl_score = (pc - oc) * 40
        pat_score = self._pattern_evaluation(board) * 25
        return dist_score + ctrl_score + pat_score

    # _calculate_min_distance, _calculate_territory_control, _pattern_evaluation,
    # _move_priority, _move_priority_opponent, get_neighbors: idénticos a LegendHexPlayer
    # para brevedad, asuma que están implementados aquí...

    def _mcts_select_move(self, board: HexBoard, simulations=100, top_k=5) -> tuple:
        """
        MCTS básico: para los top_k movimientos según heurística,
        simula `simulations` playouts aleatorios y elige el mejor.
        """
        moves = board.get_possible_moves()
        # Priorizar por heurística
        candidates = sorted(moves, key=lambda m: self._move_priority(board, m), reverse=True)[:top_k]
        wins = {m: 0 for m in candidates}

        for move in candidates:
            for _ in range(simulations):
                b2 = board.clone()
                b2.place_piece(move[0], move[1], self.player_id)
                winner = self._random_playout(b2)
                if winner == self.player_id:
                    wins[move] += 1
        # Seleccionar movimiento con más victorias
        best = max(candidates, key=lambda m: wins[m])
        return best

    def _random_playout(self, board: HexBoard) -> int:
        """Juega aleatoriamente hasta el final y retorna el ganador"""
        b = board.clone()
        players = [self.opponent_id, self.player_id]  # siguiente turno invertido
        turn = 0
        while True:
            moves = b.get_possible_moves()
            if not moves:
                return 0
            mv = random.choice(moves)
            pid = players[turn % 2]
            b.place_piece(mv[0], mv[1], pid)
            if b.check_connection(pid):
                return pid
            turn += 1

    def learn_from_game(self, result: int):
        """Actualiza move_stats con refuerzo tras terminar un juego"""
        # result: 1 si ganamos, -1 si perdimos, 0 empate
        for key, move in self.move_history:
            self.move_stats[(key, move)] = self.move_stats.get((key, move), 0) + result
        self.move_history.clear()
