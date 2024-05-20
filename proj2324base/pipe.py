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
   
        
            
    def __lt__(self, other):
        return self.id < other.id
    
    
    def __str__(self):
        return self.board.print()
   

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um board de PipeMania."""


    def __init__(self,board_input,size):
        self.board = np.array(board_input,dtype=str)
        

    def adjacent_vertical_values(self, row: int, col: int):
        """Devolve os valores imediatamente acima e abaixo, respectivamente."""
        size = self.get_size()
        above = self.board[row - 1, col] if row > 0 else None
        below = self.board[row + 1, col] if row < size - 1 else None
        return above, below


    def adjacent_horizontal_values(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita, respectivamente."""
        size = self.get_size()
        left = self.board[row, col - 1] if col > 0 else None
        right = self.board[row, col + 1] if col < size - 1 else None
        return left, right

   
    def __getitem__(self, index):
        return self.board[index]
    
    
    def get_size(self):
        return self.board.shape[0]
    
    
    def get_value(self, row: int, col: int):
        """Devolve o valor na respetiva posição do board."""
        return self.board[row, col]
    
    
    @staticmethod
    def parse_instance():
        # Read the input and strip newline character and any trailing whitespace from each line
        lines = [line.rstrip('\n') for line in stdin.readlines()]

        # Split each line into pieces by two spaces using list comprehension
        board_list = [line.split("\t") for line in lines if line]
        board = np.array(board_list)

        # Create a Board object
        board_obj = Board(board, board.shape[0])

        state = PipeManiaState(board_obj)
        
        return board_obj


    def get_piece(self, row, column):
        return self.board[row][column]
    
    
    def set_piece(self,row, col, piece):
        self.board[row][col] = piece
    
    
    def print(self):
        """Return a string representation of the board."""
        return '\n'.join('  '.join(row) for row in self.board)
    
    
    def count_real_connections(self):
        
        # Contar peças e calcular o número total de conexões esperadas
        size = len(self.board)
        connections = set()    
        for row in range(size):
            for col in range(size):
                piece = self.board[row, col]
            
                if piece == 'FB':
                    _, down = self.adjacent_vertical_values(row, col)
                    if down in {"BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        
                if piece == 'FC':
                    up, _ = self.adjacent_vertical_values(row, col)
                    if up in {"BB", "BD", "BE","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                
                if piece == 'FE':
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if left in {"BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'FD':
                    _, right = self.adjacent_horizontal_values(row, col)
                    if right in {"BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'BC':
                    up, _ = self.adjacent_vertical_values(row, col)
                    left, right = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if left in {"BC","FD","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                    if right in {"BC","FE","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'BB':
                    _, down = self.adjacent_vertical_values(row, col)
                    left, right = self.adjacent_horizontal_values(row, col)
                    if down in {"BC","FC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if left in {"BB","BC","BD","FD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'BE':
                    up, down = self.adjacent_vertical_values(row, col)
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'BD':
                    up, down = self.adjacent_vertical_values(row, col)
                    _, right = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        
                if piece == 'VC':
                    up, _ = self.adjacent_vertical_values(row, col)
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'VB':
                    _, down = self.adjacent_vertical_values(row, col)
                    _, right = self.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'VE':
                    _, down = self.adjacent_vertical_values(row, col)
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'VD':
                    up, _ = self.adjacent_vertical_values(row, col)
                    _, right = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'LH':
                    left, right = self.adjacent_horizontal_values(row, col)
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'LV':
                    up, down = self.adjacent_vertical_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
        
        return len(connections)
    # TODO: outros metodos da classe


class PipeMania(Problem):
    
    
    def __init__(self, board: Board):
        """The constructor specifies the initial state."""
        initial_state = PipeManiaState(board)
        super().__init__(initial_state)
        self.expected_connections = self.expected_connections_calc(board)   
         
         
    def actions(self, state):
        # Initialize an empty list to store the actions
        actions = []
        board = state.board  
        size = board.get_size()
        
        # For each possible rotation...
        for row in range(size):
            for col in range(size):
                piece = board[row][col]
                
                # Initialize all possible actions for this piece
                piece_actions = [(row, col, 90), (row, col, 180), (row, col, 270)]
                
                # Filter actions based on piece type and position
                if row == 0 and col == 0:  # top-left corner
                    if piece in {'FB', 'FD'}:
                        piece_actions.remove((row, col, 90))
                        piece_actions.remove((row, col, 180))
                    if piece == 'FC':
                        piece_actions.remove((row, col, 270))
                    if piece == 'FD':
                        piece_actions.remove((row, col, 270))
                    if piece == 'FE':
                        piece_actions.remove((row, col, 90))
                    if piece == 'VB':
                        piece_actions = []
                    if piece in {'VC', 'VD', 'VE'}:
                        board.set_piece(row, col, 'VB')
                        piece_actions = []

                if row == size - 1 and col == 0:  # bottom-left corner
                    if piece == 'VD':
                        piece_actions = []
                    if piece in {'VB', 'VC', 'VE'}:
                        board.set_piece(row, col, 'VD')
                        piece_actions = []

                if row == 0 and col == size - 1:  # top-right corner
                    if piece == 'VE':
                        piece_actions = []
                    if piece in {'VC', 'VB', 'VD'}:
                        board.set_piece(row, col, 'VE')
                        piece_actions = []

                if row == size - 1 and col == size - 1:  # bottom-right corner
                    if piece == 'VC':
                        piece_actions = []
                    if piece in {'VB', 'VE'}:
                        board.set_piece(row, col, 'VC')
                        piece_actions = []

                if row == 0 and 0 < col < size - 1 and size > 1:  # top row, not corners
                    if piece in {'LV', 'BB', 'BC'}:
                        board.set_piece(row, col, 'BB')
                        piece_actions = []
                    if piece == 'LH':
                        piece_actions = []

                if row == size - 1 and 0 < col < size - 1 and size > 1:  # bottom row, not corners
                    if piece in {'LV', 'BC', 'BB'}:
                        board.set_piece(row, col, 'BC')
                        piece_actions = []
                    if piece == 'LH':
                        piece_actions = []

                if col == 0 and 0 < row < size - 1 and size > 1:  # left column, not corners
                    if piece == 'LV':
                        piece_actions = []
                    if piece == 'LH':
                        board.set_piece(row, col, 'LV')
                        piece_actions = []
                    if piece == 'BD':
                        piece_actions = []
                    if piece in {'BC', 'BB', 'BE'}:
                        board.set_piece(row, col, 'BD')
                        piece_actions = []

                if col == size - 1 and 0 < row < size - 1 and size > 1:  # right column, not corners
                    if piece == 'LV':
                        piece_actions = []
                    if piece == 'LH':
                        board.set_piece(row, col, 'LV')
                        piece_actions = []
                    if piece == 'BE':
                        piece_actions = []
                    if piece in {'BC', 'BB', 'BD'}:
                        board.set_piece(row, col, 'BE')
                        piece_actions = []

                # Add valid actions for the piece to the actions list
                actions.extend(piece_actions)
        
        # Return the list of actions
        return actions

    
    def result(self, state: PipeManiaState, action):
            
        new_board = copy.deepcopy(state.board)
            
        # Unpack the action
        row, col, clockwise = action
               
        new_state = PipeManiaState(new_board)
        
        new_board[row][col] =  self.rotate_piece(new_board.get_piece(row, col), clockwise)       
        print(new_state.board.print(),"\n",sep="")
        return new_state
    
    
    def rotate_piece(self,piece, rotation):
        
        """Rotate the given piece in the specified direction."""
        if(piece == "FC"):
            if rotation == 90:
                return "FD"
            if rotation == 180:
                return "FE"
            if rotation == 270:
                return "FB"
        
        if(piece == "FD"):
            if rotation == 90:
                return "FE"
            if rotation == 180:
                return "FB"
            if rotation == 270:
                return "FC"
        
        if(piece == "FE"):
            if rotation == 90:
                return "FB"
            if rotation == 180:
                return "FC"
            if rotation == 270:
                return "FD"
        
        if(piece == "FB"):
            if rotation == 90:
                return "FC"
            if rotation == 180:
                return "FD"
            if rotation == 270:
                return "FE"
        
        if(piece == "BC"):
            if rotation == 90:
                return "BD"
            if rotation == 180:
                return "BE"
            if rotation == 270:
                return "BB"
        
        if(piece == "BD"):
            if rotation == 90:
                return "BE"
            if rotation == 180:
                return "BB"
            if rotation == 270:
                return "BC"
        
        if(piece == "BE"):
            if rotation == 90:
                return "BB"
            if rotation == 180:
                return "BC"
            if rotation == 270:
                return "BD"
        
        if(piece == "BB"):
            if rotation == 90:
                return "BC"
            if rotation == 180:
                return "BD"
            if rotation == 270:
                return "BE"
        
        if(piece == "VC"):
            if rotation == 90:
                return "VD"
            if rotation == 180:
                return "VE"
            if rotation == 270:
                return "VB"
        
        if(piece == "VD"):
            if rotation == 90:
                return "VE"
            if rotation == 180:
                return "VB"
            if rotation == 270:
                return "VC"
        
        if(piece == "VE"):
            if rotation == 90:
                return "VB"
            if rotation == 180:
                return "VC"
            if rotation == 270:
                return "VD"
        
        if(piece == "VB"):
            if rotation == 90:
                return "VC"
            if rotation == 180:
                return "VD"
            if rotation == 270:
                return "VE"
        
        if(piece == "LH"):
            if rotation == 90:
                return "LV"
            if rotation == 180:
                return "LV"
            if rotation == 270:
                return "LV"
        
        if(piece == "LV"):
            if rotation == 90:
                return "LH"
            if rotation == 180:
                return "LH"
            if rotation == 270:
                return "LH"

   
    def expected_connections_calc(self, board):
        # Contar peças e calcular o número total de conexões esperadas
        size = board.get_size()
        connections = 0
        for row in range(size):
            for col in range(size):
                piece = board.get_piece(row, col)
                if piece in {'FB', 'FC','FD','FE'}:
                    connections +=1 
                        
                if piece in {'BC', 'BB','BE','BD'}:
                    connections += 3
                    
                if piece in {'VC', 'VB','VE','VD','LH','LV'}:
                    connections += 2
        
        return int(connections/2)
    
    
    def goal_test(self, state: PipeManiaState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do board 
        estão preenchidas de acordox com as regras do problema."""
        board = state.board
        return board.count_real_connections() == self.expected_connections
    

    def h(self, node: Node):
        """Heuristic function used for A* search."""
        current_state = node.state
        #print(abs(self.expected_connections - current_state.board.count_real_connections()))
        return abs(self.expected_connections - current_state.board.count_real_connections())
    
    
    def print(self):
        return '\n'.join(' '.join(row) for row in self.board_list)
    
    # TODO: outros metodos da classe


if __name__ == "__main__":
    # Parse the instance
    board = Board.parse_instance()
    # Create a PipeMania object
    pipe_mania = PipeMania(board)
    print("initial state:\n",board.print(),sep="")
    # Solve the problem using A* search
    solution = breadth_first_tree_search(pipe_mania)
    # Print the solution
    print(solution.state.board.print())



