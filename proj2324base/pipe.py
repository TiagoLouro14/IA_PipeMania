# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 11:
# 104101 Tiago Queiroz de Orduña Dores Louro
# 103562 Guilherme Neca Ribeiro

from sys import stdin
import numpy as np
from copy import deepcopy
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

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de PipeMania."""

    def __init__(self, board_list, size:int):
        self.board = np.array(board_list)
        self.pieces = {"FC","FB","FE","FD","BC","BB",
                       "BE","BD","VC","VB","VE","VD","LH","LV"}
        self.columns = self.rows = size
          
              
    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def adjacent_vertical_values(self, row: int, col: int):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        above = self.board[row - 1][col] if row > 0 else None
        below = self.board[row + 1][col] if row < len(self.board) - 1 else None
        return above, below

    def adjacent_horizontal_values(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        left = self.board[row][col - 1] if col > 0 else None
        right = self.board[row][col + 1] if col < len(self.board[0]) - 1 else None
        return left, right

    @staticmethod
    def parse_instance():
        # Read the input and strip newline character and any trailing whitespace from each line
        lines = [line.rstrip('\n') for line in stdin.readlines()]

        # Split each line into pieces by two spaces using list comprehension
        board_list = [line.split('  ') for line in lines if line]

        return Board(board_list, len(board_list))
    
    def __str__(self):
        """Return a string representation of the board."""
        return '\n'.join('  '.join(row) for row in self.board)
    
    # TODO: outros metodos da classe

class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(board)
        self.initial = board
        pass

    def actions(self, state: PipeManiaState):
        """Return a list of actions that can be executed from the given state."""
        actions = []
        for row in range(state.board.shape[0]):
            for col in range(state.board.shape[1]):
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
        """Return the state resulting from applying the given action to the given state."""
        # Create a copy of the state
        new_state = deepcopy(state)

        # Unpack the action
        row, col, clockwise = action

        # Rotate the piece at (row, col) in the specified direction
        new_state.board.board[row][col] = self.rotate_piece(new_state.board.board[row][col], clockwise)

        return new_state

    def goal_test(self, piece, clockwise):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass
    
    def needs_rotation(self, piece):
        """Return True if the given piece needs to be rotated to match the goal."""
        if(piece == "FC" or piece == "BC" or piece == "VC"):
            return True
        return False

    def h(self, node: Node):
        """Heuristic function used for A* search."""
        # Get the current state
        state = node.state

        # Initialize count of pieces that need to be rotated
        count = 0

        # Iterate over each piece in the board
        for row in range(state.board.shape[0]):
            for col in range(state.board.shape[1]):
                # If the piece needs to be rotated, increment the count
                if self.needs_rotation(state.board[row][col]):
                    count += 1

        return count

    def __str__(self):
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

# Ler grelha do figura 1a:
board = Board.parse_instance()
print(board)

problem = PipeMania(board)
# Criar um estado com a configuração inicial:
s0 = PipeManiaState(board)
# Aplicar as ações que resolvem a instância
s1 = problem.result(s0, (0, 1, True))
s2 = problem.result(s1, (0, 1, True))
s3 = problem.result(s2, (0, 2, True))
s4 = problem.result(s3, (0, 2, True))
s5 = problem.result(s4, (1, 0, True))
s6 = problem.result(s5, (1, 1, True))
s7 = problem.result(s6, (2, 0, False)) # anti-clockwise (exemplo de uso) 
s8 = problem.result(s7, (2, 0, False)) # anti-clockwise (exemplo de uso) 
s9 = problem.result(s8, (2, 1, True))
s10 = problem.result(s9, (2, 1, True))
s11 = problem.result(s10, (2, 2, True))

print("\nsolution?:\n")
print(s11.board)