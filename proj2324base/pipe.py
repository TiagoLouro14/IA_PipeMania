# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 11:
# 104101 Tiago Queiroz de Orduña Dores Louro
# 103562 Guilherme Neca Ribeiro

from sys import stdin
import numpy as np
import copy
from search import Problem
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class PipeManiaState:
    state_id = 0

    
    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1
        self.real_count = self.count_real_connections(board) * 2
       
        
    def __lt__(self, other):
        return self.id < other.id
    
    
    def count_real_connections(self,board):
        # Contar peças e calcular o número total de conexões esperadas
        size = board.get_size()
        
        connections = set()
        
        for row in range(size):
            for col in range(size):
                piece = self.board.get_piece(row, col)
            
                if piece == 'FB':
                    _, down = board.adjacent_vertical_values(row, col)
                    if down in {"BC","BE","BD","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
                        
                if piece == 'FC':
                    up, _ = board.adjacent_vertical_values(row, col)
                    if up in {"BB", "BD", "BE","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                
                if piece == 'FE':
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if left in {"BC","BB","BD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                
                if piece == 'FD':
                    _, right = board.adjacent_horizontal_values(row, col)
                    if right in {"BC","BB","BE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                
                if piece == 'BC':
                    up, _ = board.adjacent_vertical_values(row, col)
                    left, right = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                    if left in {"BC","FD","BB","BD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                    if right in {"BC","FE","BB","BE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                
                if piece == 'BB':
                    _, down = board.adjacent_vertical_values(row, col)
                    left, right = board.adjacent_horizontal_values(row, col)
                    if down in {"BC","FC","BE","BD","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
                    if left in {"BB","BC","BD","FD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                
                if piece == 'BE':
                    up, down = board.adjacent_vertical_values(row, col)
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                
                if piece == 'BD':
                    up, down = board.adjacent_vertical_values(row, col)
                    _, right = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                        
                if piece == 'VC':
                    up, _ = board.adjacent_vertical_values(row, col)
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                
                if piece == 'VB':
                    _, down = board.adjacent_vertical_values(row, col)
                    _, right = board.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                
                if piece == 'VE':
                    _, down = board.adjacent_vertical_values(row, col)
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                
                if piece == 'VD':
                    up, _ = board.adjacent_vertical_values(row, col)
                    _, right = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                
                if piece == 'LH':
                    left, right = board.adjacent_horizontal_values(row, col)
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(((row, col), (row, col-1)))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(((row, col), (row, col+1)))
                
                if piece == 'LV':
                    up, down = board.adjacent_vertical_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(((row, col), (row-1, col)))
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(((row, col), (row+1, col)))
        
          
        unique_connections = {frozenset(conn) for conn in connections}
        return len(unique_connections)

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""


    def __init__(self,board_input,size):
        self.board = np.array(board_input,dtype=str)


    def adjacent_vertical_values(self, row: int, col: int):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        above = self.board[row - 1][col] if row > 0 else None
        below = self.board[row + 1][col] if row < self.board.shape[0] - 1 else None
        return above, below


    def adjacent_horizontal_values(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left = self.board[row][col - 1] if col > 0 else None
        right = self.board[row][col + 1] if col < self.board.shape[1] - 1 else None
        return left, right


    def get_size(self):
        return self.board.shape[0]
    
    
    def get_value(self, row: int, col: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]
    
    
    @staticmethod
    def parse_instance():
        # Read the input and strip newline character and any trailing whitespace from each line
        lines = [line.rstrip('\n') for line in stdin.readlines()]

        # Split each line into pieces by two spaces using list comprehension
        board_list = [line.split('  ') for line in lines if line]
        board = np.array(board_list)

        # Create a Board object
        board_obj = Board(board, board.shape[0])

        return board_obj


    def get_piece(self, row, column):
        return self.board[row][column]
    
    
    def set_piece(self, row, col, piece):
        self.board[row][col] = piece
    
    
    def print(self):
        """Return a string representation of the board."""
        return '\n'.join('  '.join(row) for row in self.board)
    

    # TODO: outros metodos da classe


class PipeMania(Problem):
    
    
    def __init__(self, board: Board):
        """The constructor specifies the initial state."""
        self.board = board
        initial_state = PipeManiaState(board)
        super().__init__(initial_state)
        self.expected_count = self.expected_connections(board)


    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        size = board.get_size()
        actions = []
        for row in range(size):
            for col in range(size):
                # Add actions to rotate the piece at (row, col) in both directions
                actions.append((row, col, True))
                actions.append((row, col, False))
        return actions
        
    
    def rotate_piece(self,piece, clockwise):
        
        """Rotate the given piece in the specified direction."""
        if(piece == "FC"):
            return "FD" if clockwise else "FE"
        
        if(piece == "FD"):
            return "FB" if clockwise else "FC"
        
        if(piece == "FB"):
            return "FE" if clockwise else "FD"
        
        if(piece == "FE"):
            return "FC" if clockwise else "FB"       
        
        if(piece == "BC"):
            return "BD" if clockwise else "BE"
        
        if(piece == "BD"):
            return "BB" if clockwise else "BC" 
        
        if(piece == "BB"):
            return "BE" if clockwise else "BD"
        
        if(piece == "BE"):
            return "BC" if clockwise else "BB"
        
        if(piece == "VC"):
            return "VD" if clockwise else "VE"
        
        if(piece == "VD"):
            return "VB" if clockwise else "VC"
        
        if(piece == "VB"):
            return "VE" if clockwise else "VD"
        
        if(piece == "VE"):
            return "VC" if clockwise else "VB"
        
        if(piece == "LH"):
            return "LV" if clockwise else "LH"
        
        if(piece == "LV"):
            return "LH" if clockwise else "LV"
    
    
    def result(self, state: PipeManiaState, action):
        # Create a deep copy of the current state's board
        new_board = copy.deepcopy(state.board)

        # Unpack the action
        row, col, clockwise = action

        # Get the piece at the specified coordinates
        piece = new_board.get_piece(row, col)

        # Rotate the piece
        rotated_piece = self.rotate_piece(piece, clockwise)

        # Set the rotated piece at the specified coordinates
        new_board.set_piece(row, col, rotated_piece)

        # Create a new state with the updated board
        new_state = PipeManiaState(new_board)
    
        return new_state
    
    
    def expected_connections(self, board):
        """Calcula o número de conexões esperadas para o tabuleiro dado sem duplicação."""
        size = board.get_size()
        connections = set()
        
        # Definição das direções para facilitar a verificação de conexões
        directions = {
            'C': (-1, 0),  # up
            'B': (1, 0),   # down
            'E': (0, -1),  # left
            'D': (0, 1)    # right
        }
        
        def is_valid_position(x, y):
            return 0 <= x < size and 0 <= y < size

        for row in range(size):
            for col in range(size):
                piece = board.get_piece(row, col)
                
                if piece.startswith('F'):
                    dir = piece[1]
                    dr, dc = directions[dir]
                    nr, nc = row + dr, col + dc
                    if is_valid_position(nr, nc):
                        connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'BC':
                    for dir in ['C', 'E', 'D']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'BB':
                    for dir in ['B', 'E', 'D']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'BE':
                    for dir in ['C', 'B', 'E']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'BD':
                    for dir in ['C', 'B', 'D']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'VC':
                    for dir in ['C', 'E']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'VB':
                    for dir in ['B', 'D']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'VE':
                    for dir in ['B', 'E']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'VD':
                    for dir in ['C', 'D']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'LH':
                    for dir in ['E', 'D']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

                elif piece == 'LV':
                    for dir in ['C', 'B']:
                        dr, dc = directions[dir]
                        nr, nc = row + dr, col + dc
                        if is_valid_position(nr, nc):
                            connections.add(frozenset([(row, col), (nr, nc)]))

        return len(connections)


    def goal_test(self, state):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas de acordo com as regras do problema."""
        return self.expected_connections(board) == state.real_count
    

    def h(self, node: Node):
        """Heuristic function used for A* search."""
        
        return 1
    
    
    def print(self):
        return '\n'.join(' '.join(row) for row in self.board_list)
    
    # TODO: outros metodos da classe

'''
if __name__ == "__main__":
    # Read the input file from standard input
    board = Board.parse_instance()

    # Create an instance of the PipeMania problem
    problem = PipeMania(board)

    # Use a search technique to solve the instance
    solution_node = astar_search(problem)

    # Extract the solution from the resulting node
    solution = solution_node.solution()

    # Print the solution to the standard output in the specified format
    print(solution)

'''

