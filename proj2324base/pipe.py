# pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 11:
# 104101 Tiago Queiroz de Orduña Dores Louro


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
        pieces_connections = {}   
        for row in range(size):
            for col in range(size):
                piece = self.board[row, col]
                if piece not in pieces_connections:
                    pieces_connections[piece] = 0
                if piece == 'FB':
                    _, down = self.adjacent_vertical_values(row, col)
                    if down in {"BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
                        
                        
                if piece == 'FC':
                    up, _ = self.adjacent_vertical_values(row, col)
                    if up in {"BB", "BD", "BE","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                
                if piece == 'FE':
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if left in {"BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'FD':
                    _, right = self.adjacent_horizontal_values(row, col)
                    if right in {"BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'BC':
                    up, _ = self.adjacent_vertical_values(row, col)
                    left, right = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                    if left in {"BC","FD","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                    if right in {"BC","FE","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'BB':
                    _, down = self.adjacent_vertical_values(row, col)
                    left, right = self.adjacent_horizontal_values(row, col)
                    if down in {"BC","FC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
                    if left in {"BB","BC","BD","FD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'BE':
                    up, down = self.adjacent_vertical_values(row, col)
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'BD':
                    up, down = self.adjacent_vertical_values(row, col)
                    _, right = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BE","BB","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                    if down in {"FC","BC","BD","BE","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
                    if right in {"BB","BC","BE","FE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                        
                if piece == 'VC':
                    up, _ = self.adjacent_vertical_values(row, col)
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'VB':
                    _, down = self.adjacent_vertical_values(row, col)
                    _, right = self.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'VE':
                    _, down = self.adjacent_vertical_values(row, col)
                    left, _ = self.adjacent_horizontal_values(row, col)
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'VD':
                    up, _ = self.adjacent_vertical_values(row, col)
                    _, right = self.adjacent_horizontal_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'LH':
                    left, right = self.adjacent_horizontal_values(row, col)
                    if left in {"FD","BC","BB","BD","VB","VD","LH"}:
                        connections.add(frozenset((((row, col), (row, col-1)))))
                        pieces_connections[piece] += 1
                    if right in {"FE","BC","BB","BE","VC","VE","LH"}:
                        connections.add(frozenset((((row, col), (row, col+1)))))
                        pieces_connections[piece] += 1
                
                if piece == 'LV':
                    up, down = self.adjacent_vertical_values(row, col)
                    if up in {"FB","BB","BE","BD","VB","VE","LV"}:
                        connections.add(frozenset((((row, col), (row-1, col)))))
                        pieces_connections[piece] += 1
                    if down in {"FC","BC","BE","BD","VC","VD","LV"}:
                        connections.add(frozenset((((row, col), (row+1, col)))))
                        pieces_connections[piece] += 1
        
        return len(connections), [(piece, count) for piece, count in pieces_connections.items()]
    


class PipeMania(Problem):
    
    def __init__(self, board):
        initial_state = PipeManiaState(board)
        super().__init__(initial_state)
        self.expected_connections = self.expected_connections_calc(board)   
         
        
        
    def count_unconnected_pipes(self, board):
        state = PipeManiaState(board)
        board = state.board
        unconnected_pipes = 0
        # Get the real connections for each piece
        _, connections = board.count_real_connections()
        
        for piece, count in connections:
            if piece in {'FB', 'FC','FD','FE'}:
                if count < 1:
                    unconnected_pipes += 1
            if piece in {'BC', 'BB','BE','BD'}:
                if count < 3:
                    unconnected_pipes += 1
            if piece in {'VC', 'VB','VE','VD','LH','LV'}:
                if count < 2:
                    unconnected_pipes += 1
                    
        return unconnected_pipes

    
         
    def actions(self, state):
        # Initialize an empty list to store the actions
        actions = []
        board = state.board
        size = board.get_size()
        
        # Use a set for actions for faster membership checking
        middle_actions = set()
        
        # middle pieces
        transformations = {
            'VB': {'VC', 'VD', 'VE'},
            'VC': {'VB', 'VD', 'VE'},
            'VD': {'VB', 'VC', 'VE'},
            'VE': {'VB', 'VC', 'VD'},
            'FB': {'FC', 'FD', 'FE'},
            'FC': {'FB', 'FD', 'FE'},
            'FD': {'FB', 'FC', 'FE'},
            'FE': {'FB', 'FC', 'FD'},
            'BB': {'BC', 'BD', 'BE'},
            'BC': {'BB', 'BD', 'BE'},
            'BD': {'BB', 'BC', 'BE'},
            'BE': {'BB', 'BC', 'BD'},
            'LH': {'LV'},
            'LV': {'LH'}
        }
          
        # For each possible rotation...
        for row in range(size):
            for col in range(size):
                
                # Get the piece in the current position
                piece = board[row][col]
                
                # Filter actions based on piece type and position
                
                # top-left corner
                if row == 0 and col == 0:  
                    if piece == 'FB':
                        board.set_piece(row, col, 'FD')
                    elif piece == 'FD':
                        board.set_piece(row, col, 'FB')
                    elif piece == 'FC':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))                    
                    elif piece in {'VC', 'VD', 'VE'}:
                        board.set_piece(row, col, 'VB')  

                # bottom-left corner
                elif row == size - 1 and col == 0:
                    if piece == 'FC':
                        board.set_piece(row, col, 'FD')
                    elif piece == 'FD':
                        board.set_piece(row, col, 'FC')
                    elif piece == 'FE':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'FB':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                    elif piece in {'VB', 'VC', 'VE'}:
                        board.set_piece(row, col, 'VD')
                    
                # top-right corner
                elif row == 0 and col == size - 1:  
                    if piece == 'FB':
                        board.set_piece(row, col, 'FE')
                    elif piece == 'FE':
                        board.set_piece(row, col, 'FB')
                    elif piece == 'FC':
                        actions.append((row, col, 'FB'))
                        actions.append((row, col, 'FE'))
                    elif piece == 'FD':
                        actions.append((row, col, 'FB'))
                        actions.append((row, col, 'FE'))  
                    elif piece in {'VC', 'VB', 'VD'}:
                        board.set_piece(row, col, 'VE')
                      
                # bottom-right corner
                elif row == size - 1 and col == size - 1: 
                    if piece == 'FC':
                        board.set_piece(row, col, 'FE')
                    elif piece == 'FE':
                        board.set_piece(row, col, 'FC')
                    elif piece == 'FD':
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'FB':
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FE')) 
                    elif piece in {'VB', 'VE','VD'}:
                        board.set_piece(row, col, 'VC')
                        
                # top row, not corners
                elif row == 0 and 0 < col < size - 1 and size > 2:
                    if piece in {'VD','VC'}:
                        actions.append((row, col, 'VB'))
                        actions.append((row, col, 'VE'))
                    elif piece == 'VB':
                        board.set_piece(row, col, 'VC')
                    elif piece == 'VE':
                        board.set_piece(row, col, 'VD')
                    elif piece == 'FC':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))
                        actions.append((row, col, 'FE'))
                    elif piece in {'FD', 'FB', 'FE'}:
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))
                        actions.append((row, col, 'FE'))
                    elif piece in {'BD', 'BE', 'BC'}:
                        board.set_piece(row, col, 'BB')
                    elif piece == 'LV':
                        board.set_piece(row, col, 'LH')
                        
                # bottom row, not corners
                elif row == size - 1 and 0 < col < size - 1 and size > 2:
                    if piece in {'VE','VB'}:
                        actions.append((row, col, 'VC'))
                        actions.append((row, col, 'VD'))
                    elif piece == 'VC':
                        board.set_piece(row, col, 'VD')
                    elif piece == 'VD':
                        board.set_piece(row, col, 'VC')
                    elif piece == 'FB':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FE'))
                    elif piece in {'FD', 'FC', 'FE'}:
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FE'))   
                    elif piece in {'BD', 'BE', 'BB'}:
                        board.set_piece(row, col, 'BC')
                    elif piece == 'LV':
                        board.set_piece(row, col, 'LH')
                        
                # left side, not corners
                elif col == 0 and 0 < row < size - 1 and size > 2:
                    if piece in {'VC','VE'}:
                        actions.append((row, col, 'VB'))
                        actions.append((row, col, 'VD'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FB'))
                    elif piece in {'FD', 'FB', 'FC'}:
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'LH':
                        board.set_piece(row, col, 'LV')
                    elif piece in {'BC', 'BB', 'BE'}:
                        board.set_piece(row, col, 'BD')
                        
                # right side, not corners
                elif col == size - 1 and 0 < row < size - 1 and size > 2:
                    if piece in {'VD','VB'}:
                        actions.append((row, col, 'VC'))
                        actions.append((row, col, 'VE'))  
                    elif piece == 'FD':
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FB'))
                    elif piece in {'FE', 'FC', 'FB'}:
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FB'))
                    elif piece == 'LH':
                        board.set_piece(row, col, 'LV')
                    elif piece in {'BC', 'BB', 'BD'}:
                        board.set_piece(row, col, 'BE')

                
                elif 0 < row < size - 1 and 0 < col < size - 1 and size > 2:
                    if piece in transformations:
                        for new_piece in transformations[piece]:
                            action = (row, col, new_piece)
                            middle_actions.add(action)  # Add to the intermediate set
                            
        # Extend the actions list with the intermediate actions
        actions.extend(list(middle_actions))                     
        print(actions)
        
        #Return the list of actions
        return actions

    
    def result(self, state: PipeManiaState, action):
            
        new_board = copy.deepcopy(state.board)
            
        # Unpack the action
        row, col, piece = action
        
        #print("antes:\n",state.board.print(),"\n",sep="")
        new_board.set_piece(row, col, piece)
        
        new_state = PipeManiaState(new_board)
        
        
        #print("depois;\n",new_state.board.print(),"\n",sep="")
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
        board = state.board
        return board.count_real_connections()[0] == self.expected_connections
    
    
    def h(self, node: Node):
        """Heuristic function used for A* search."""
        current_state = node.state
        # Count the number of unconnected pipes
        unconnected_pipes = self.count_unconnected_pipes(current_state.board)
        # Calculate the expected number of connections when the board is complete
        expected_connections = self.expected_connections_calc(current_state.board)
        
        # Count the current number of real connections on the board
        real_connections = current_state.board.count_real_connections()[0]
        
        # Heuristic is the sum of the difference in connections and the number of unconnected pipes
        heuristic_value = abs(real_connections - expected_connections) + unconnected_pipes
        # Debug print to check heuristic value
        #print("h:", heuristic_value)
        
        return heuristic_value
 
    
    def print(self):
        return '\n'.join(' '.join(row) for row in self.board_list)
    
    # TODO: outros metodos da classe


if __name__ == "__main__":
    # Parse the instance
    board = Board.parse_instance()
    # Create a PipeMania object
    pipe_mania = PipeMania(board)
    # Solve the problem using A* search
    solution = astar_search(pipe_mania)
    # Print the solution
    print(solution.state.board.print())
    



