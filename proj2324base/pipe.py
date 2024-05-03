# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 11:
# 104101 Tiago Queiroz de Orduña Dores Louro
# 103562 Guilherme Neca Ribeiro

from sys import stdin
import numpy as np
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

    def __init__(self, board):
        self.board = board
        self.pieces = {"FC","FB","FE","FD",
        BC,BB,BE,BD,VC,VB,VE,VD,LH,LV}
        self.columns = self.rows = len(board)  
              
    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.board[row][col]

    def adjacent_vertical_values(self, row:int, col:int):
        above = self.board[row-1][col] if row > 0 else None
        below = self.board[row+1][col] if row+1 < len(self.board) else None
        return (above, below)

    def adjacent_horizontal_values(self, row:int, col:int):
        left = self.board[row][col-1] if col > 0 else None
        right = self.board[row][col+1] if col+1 < len(self.board[row]) else None
        return (left, right)

    @staticmethod
    def parse_instance():
        board_list = []
        while True:
            line = stdin.readline().split()
            if not line:  
                break
            board_list.append(line)
        return Board(board_list) 
    
    # TODO: outros metodos da classe

class PipeMania(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(board)
        self.initial = board
        pass

    def actions(self, state: PipeManiaState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: PipeManiaState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

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

print(board.adjacent_vertical_values(0, 0))
print(board.adjacent_horizontal_values(0, 0))
print(board.adjacent_vertical_values(1, 1))
print(board.adjacent_horizontal_values(1, 1))
