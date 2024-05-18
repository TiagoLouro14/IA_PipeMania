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
        self.real_count = self.count_real_connections(board)
        
        
          
    def __lt__(self, other):
        return self.id < other.id
    
    
    def __str__(self):
        return self.board.print()
   
   
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
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        
                if piece == 'FC':
                    up, _ = board.adjacent_vertical_values(row, col)
                    if up in {"BB", "BD", "BE","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                
                if piece == 'FE':
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if left in {"BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'FD':
                    _, right = board.adjacent_horizontal_values(row, col)
                    if right in {"BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'BC':
                    up, _ = board.adjacent_vertical_values(row, col)
                    left, right = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if left in {"BC","FD","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                    if right in {"BC","FE","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'BB':
                    _, down = board.adjacent_vertical_values(row, col)
                    left, right = board.adjacent_horizontal_values(row, col)
                    if down in {"BC","FC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if left in {"BB","BC","BD","FD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'BE':
                    up, down = board.adjacent_vertical_values(row, col)
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'BD':
                    up, down = board.adjacent_vertical_values(row, col)
                    _, right = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        
                if piece == 'VC':
                    up, _ = board.adjacent_vertical_values(row, col)
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'VB':
                    _, down = board.adjacent_vertical_values(row, col)
                    _, right = board.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'VE':
                    _, down = board.adjacent_vertical_values(row, col)
                    left, _ = board.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                
                if piece == 'VD':
                    up, _ = board.adjacent_vertical_values(row, col)
                    _, right = board.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'LH':
                    left, right = board.adjacent_horizontal_values(row, col)
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                
                if piece == 'LV':
                    up, down = board.adjacent_vertical_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
        #print(connections)
        return len(connections)


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


    def get_size(self):
        return self.board.shape[0]
    
    
    def get_value(self, row: int, col: int):
        """Devolve o valor na respetiva posição do board."""
        return self.board[row][col]
    
    
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
        self.expected_connections = self.expected_connections_calc(board)
        self.visited_states = set()
        
         
    def actions(self, state):
        # Initialize an empty list to store the actions
        actions = []
        # For each piece on the board...
        for row in range(state.board.get_size()):
            for col in range(state.board.get_size()):
                # For each possible rotation...
                for clockwise in [True, False]:
                    # Add the action to the list
                    actions.append((row, col, clockwise))
        # Return the list of actions
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
            return "VB" if clockwise else "VE"
        
        if(piece == "VD"):
            return "VB" if clockwise else "VC"
        
        if(piece == "VB"):
            return "VE" if clockwise else "VD"
        
        if(piece == "VE"):
            return "VC" if clockwise else "VB"
        
        if(piece == "LH"):
            return "LV" if clockwise else "LV"
        
        if(piece == "LV"):
            return "LH" if clockwise else "LH"
    

    def result(self, state: PipeManiaState, action):
        # Unpack the action
        row, col, clockwise = action
        
        # Create a new instance of PipeManiaState with a deep copy of the board
        new_state = PipeManiaState(board=copy.deepcopy(state.board))
        
        if new_state.id in self.visited_states:
            return None
        
        # Increment the id of the old state
        state.id += 1

        # Set the id of the new state
        new_state.id = state.id

        # Copy the real_count from the old state to the new state
        new_state.real_count = state.real_count

        # Get the piece at the specified coordinates
        piece = new_state.board.get_piece(row, col)

        # Rotate the piece
        rotated_piece = self.rotate_piece(piece, clockwise)

        # Set the rotated piece at the specified coordinates
        new_state.board.set_piece(row, col, rotated_piece)

        # Update real_count
        new_state.real_count = new_state.count_real_connections(new_state.board)

        # Update self.board
        self.board = copy.deepcopy(new_state.board)

        self.visited_states.add(new_state)
        #print("r",new_state.id,"result",new_state.real_count,"\n",sep="")
        return new_state
   
   
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
        return state.real_count == self.expected_connections
    

    def h(self, node: Node):
        """Heuristic function used for A* search."""
        current_state = node.state
        return abs(self.expected_connections - current_state.real_count)
    
    
    def print(self):
        return '\n'.join(' '.join(row) for row in self.board_list)
    
    # TODO: outros metodos da classe


if __name__ == "__main__":
    # Parse the instance
    board = Board.parse_instance()
    # Create a PipeMania object
    pipe_mania = PipeMania(board)
    # Solve the problem using A* search
    solution = breadth_first_tree_search(pipe_mania)
    # Print the solution
    print(solution.solution())
    print(solution.path_cost)

