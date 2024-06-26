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
      return '\n'.join('\t'.join(piece if j != len(row) - 1 else piece.rstrip() for j, piece in enumerate(row)) for row in self.board)
    
    
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
        self.restrictions(initial_state)
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


    def restrictions(self, state):
        board = state.board
        size = board.get_size()
        
        # canto superior esquerdo 
        
        #bode expiatorio
        if board[0][0] in {'FC','FE'}:
            board.set_piece(0, 0, 'FB')
            
        if board[0][0].startswith('V'):
            board.set_piece(0, 0, 'VB')
           
        # canto superior direito
        
        #bode expiatorio
        if board[0][size-1] in {'FC','FD'}:
            board.set_piece(0, size-1, 'FE')
           
        if board[0][size-1].startswith('V'):
            board.set_piece(0, size-1, 'VE')
        
        # canto inferior esquerdo
        
        #bode expiatorio
        if board[size-1][0] in {'FB','FE'}:
            board.set_piece(size-1, 0, 'FC')
            
        if board[size-1][0].startswith('V'):
            board.set_piece(size-1, 0, 'VD')
        
        # canto inferior direito
        
        #bode expiatorio
        if board[size-1][size-1] in {'FB','FD'}:
            board.set_piece(size-1, size-1, 'FC')
            
        if board[size-1][size-1].startswith('V'):
            board.set_piece(size-1, size-1, 'VC')
            
        for row in range(size):
            for col in range(size):
                
                # em cima 
                if row == 0 and 0 < col < size - 1 and size > 2:
                    if board[row][col].startswith('L'):
                        board.set_piece(row, col, 'LH')
                        
                    elif board[row][col] == 'VD' or board[row][col] == 'VC':
                        board.set_piece(row, col, 'VB')
                    
                    elif board[row][col].startswith('B'):
                        board.set_piece(row, col, 'BB')
                    #bode expiatorio
                    elif board[row][col] == 'FC':
                        board.set_piece(row, col, 'FB')
                        
                # em baixo
                elif row == size - 1 and 0 < col < size - 1 and size > 2:
                    if board[row][col].startswith('L'):
                        board.set_piece(row, col, 'LH')
                    
                    elif board[row][col] == 'VB' or board[row][col] == 'VE':
                        board.set_piece(row, col, 'VC')
                    
                    elif board[row][col].startswith('B'):
                        board.set_piece(row, col, 'BC') 
                        
                    elif board[row][col] == 'FB':
                        board.set_piece(row, col, 'FE')
                        
                # esquerda
                elif col == 0 and 0 < row < size - 1 and size > 2:
                    if board[row][col].startswith('L'):
                        board.set_piece(row, col, 'LV')
                    
                    elif board[row][col] == 'VC' or board[row][col] == 'VE':
                        board.set_piece(row, col, 'VB')
                    
                    elif board[row][col].startswith('B'):
                        board.set_piece(row, col, 'BD')
                        
                    #bode expiatorio
                    elif board[row][col] == 'FE':
                        board.set_piece(row, col, 'FB')
                
                # direita
                elif col == size - 1 and 0 < row < size - 1 and size > 2:
                    if board[row][col].startswith('L'):
                        board.set_piece(row, col, 'LV')
                    
                    elif board[row][col] == 'VB' or board[row][col] == 'VD':
                        board.set_piece(row, col, 'VC')
                    
                    elif board[row][col].startswith('B'):
                        board.set_piece(row, col, 'BE')

                    #bode expiatorio
                    elif board[row][col] == 'FD':
                        board.set_piece(row, col, 'FE')
     

    def actions(self, state):
      
        # Initialize an empty list to store the actions
        actions = []
        board = state.board
        size = board.get_size()
        
        # For each possible rotation...
        for row in range(size):
            for col in range(size):
                # Get the piece in the current position
                piece = board[row][col]
                up, down = board.adjacent_vertical_values(row, col)
                left, right = board.adjacent_horizontal_values(row, col)
                # top-left corner
                if row == 0 and col == 0:  
                    if piece == 'FB':
                        actions.append((row, col, 'FD'))
                    elif piece == 'FD':
                        actions.append((row, col, 'FB'))               

                # bottom-left corner
                elif row == size - 1 and col == 0:
                    if piece == 'FC':
                        actions.append((row, col, 'FD'))
                    elif piece == 'FD':
                        actions.append((row, col, 'FC'))
                    
                # top-right corner
                elif row == 0 and col == size - 1:  
                    if piece == 'FB':
                        actions.append((row, col, 'FE'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FB'))
                        
                # bottom-right corner
                elif row == size - 1 and col == size - 1: 
                    if piece == 'FC':
                        actions.append((row, col, 'FE'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FC'))
                        
                        
                # top row, not corners
                elif row == 0 and 0 < col < size - 1 and size > 2:
                    if piece == 'VB':
                        actions.append((row, col, 'VE'))
                    elif piece == 'VE':
                        actions.append((row, col, 'VB'))
                    elif piece == 'FD':
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FB'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))
                    elif piece =='FB':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FE'))
                        
                # bottom row, not corners
                elif row == size - 1 and 0 < col < size - 1 and size > 2:
                    if piece == 'VC':
                        actions.append((row, col, 'VD'))
                    elif piece == 'VD':
                        actions.append((row, col, 'VC'))
                    elif piece == 'FD':
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'FC':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FE')) 
                        
                # left side, not corners
                elif col == 0 and 0 < row < size - 1 and size > 2:
                    if piece == 'VB':
                        actions.append((row, col, 'VD'))
                    elif piece == 'VD':
                        actions.append((row, col, 'VB'))
                    elif piece == 'FB':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'FD':
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FB'))
                    elif piece == 'FC':
                        actions.append((row, col, 'FD'))
                        actions.append((row, col, 'FB'))
                        
                # right side, not corners
                elif col == size - 1 and 0 < row < size - 1 and size > 2:
                    if piece == 'VC':
                        actions.append((row, col, 'VE'))
                    elif piece == 'VE':
                        actions.append((row, col, 'VC'))
                    elif piece == 'FB':
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FC'))
                    elif piece == 'FC':
                        actions.append((row, col, 'FE'))
                        actions.append((row, col, 'FB'))
                    elif piece == 'FE':
                        actions.append((row, col, 'FC'))
                        actions.append((row, col, 'FB'))
                    
                # middle pieces
                if 0 < row < size - 1 and 0 < col < size - 1 and size > 2:
                    if piece.startswith('B'):
                        left_condition = left in {'FD','BC','BB','BD','VB','VD','LH'}
                        right_condition = right in {'FE','BC','BB','BE','VC','VE','LH'}
                        up_condition = up in {'FB','BE','BD','VB','VE','VD','LV'}
                        down_condition = down in {'FC','BE','BD','VB','VE','VD','LV'}
                        
                        if piece == 'BB':
                            actions_bb = set()
                            if left_condition:
                                actions_bb.add((row, col, 'BD'))
                                actions_bb.add((row, col, 'BC'))
                                if down_condition:
                                    actions_bb.add((row, col, 'BD'))
                            if right_condition:
                                actions_bb.add((row, col, 'BE'))
                                actions_bb.add((row, col, 'BC'))
                                if down_condition:
                                    actions_bb.add((row, col, 'BE'))
                            if down_condition:
                                actions_bb.add((row, col, 'BE'))
                                actions_bb.add((row, col, 'BD'))
                                if left_condition:
                                    actions_bb.add((row, col, 'BD'))
                            actions.extend(actions_bb)
                         
                        if piece == 'BD':
                            actions_bd = set()
                            if left_condition:
                                actions_bd.add((row, col, 'BE'))
                                actions_bd.add((row, col, 'BC'))
                                if down_condition:
                                    actions_bd.add((row, col, 'BE'))
                            if right_condition:
                                actions_bd.add((row, col, 'BB'))
                                actions_bd.add((row, col, 'BC'))
                                if down_condition:
                                    actions_bd.add((row, col, 'BB'))
                            if down_condition:
                                actions_bd.add((row, col, 'BB'))
                                actions_bd.add((row, col, 'BE'))
                                if left_condition:
                                    actions_bd.add((row, col, 'BE'))
                            actions.extend(actions_bd)
                        
                        if piece == 'BC':
                            actions_bc = set()
                            if left_condition:
                                actions_bc.add((row, col, 'BD'))
                                actions_bc.add((row, col, 'BB'))
                                if up_condition:
                                    actions_bc.add((row, col, 'BD'))
                            if right_condition:
                                actions_bc.add((row, col, 'BE'))
                                actions_bc.add((row, col, 'BB'))
                                if up_condition:
                                    actions_bc.add((row, col, 'BE'))
                            if up_condition:
                                actions_bc.add((row, col, 'BE'))
                                actions_bc.add((row, col, 'BD'))
                                if left_condition:
                                    actions_bc.add((row, col, 'BD'))
                            actions.extend(actions_bc)
                        
                        if piece == 'BE':
                            actions_be = set()
                            if left_condition:
                                actions_be.add((row, col, 'BD'))
                                actions_be.add((row, col, 'BB'))
                                if up_condition:
                                    actions_be.add((row, col, 'BD'))
                            if right_condition:
                                actions_be.add((row, col, 'BC'))
                                actions_be.add((row, col, 'BB'))
                                if up_condition:
                                    actions_be.add((row, col, 'BC'))
                            if up_condition:
                                actions_be.add((row, col, 'BC'))
                                actions_be.add((row, col, 'BD'))
                                if left_condition:
                                    actions_be.add((row, col, 'BD'))
                            actions.extend(actions_be)
                    if piece.startswith('V'):
                        left_condition = left in {'FD','BC','BB','BD','VB','VD','LH'}
                        right_condition = right in {'FE','BC','BB','BE','VC','VE','LH'}
                        up_condition = up in {'FB','BE','BD','VB','VE','VD','LV'}
                        down_condition = down in {'FC','BE','BD','VB','VE','VD','LV'}
                        
                        if piece == 'VB':
                            actions_vb = set()
                            if left_condition:
                                actions_vb.add((row, col, 'VD'))
                                actions_vb.add((row, col, 'VC'))
                                if up_condition:
                                    actions_vb.add((row, col, 'VD'))
                            if right_condition:
                                actions_vb.add((row, col, 'VE'))
                                actions_vb.add((row, col, 'VC'))
                                if up_condition:
                                    actions_vb.add((row, col, 'VE'))
                            if up_condition:
                                actions_vb.add((row, col, 'VE'))
                                actions_vb.add((row, col, 'VD'))
                                if left_condition:
                                    actions_vb.add((row, col, 'VD'))
                            actions.extend(actions_vb)
                            
                                    
                        if piece == 'VC':
                            actions_vc = set()
                            if left_condition:
                                actions_vc.add((row, col, 'VD'))
                                actions_vc.add((row, col, 'VB'))
                                if down_condition:
                                    actions_vc.add((row, col, 'VD'))
                            if right_condition:
                                actions_vc.add((row, col, 'VE'))
                                actions_vc.add((row, col, 'VB'))
                                if down_condition:
                                    actions_vc.add((row, col, 'VE'))
                            if down_condition:
                                actions_vc.add((row, col, 'VE'))
                                actions_vc.add((row, col, 'VD'))
                                if left_condition:
                                    actions_vc.add((row, col, 'VD'))
                            actions.extend(actions_vc)
                            
                        if piece == 'VE':
                            actions_ve = set()
                            if left_condition:
                                actions_ve.add((row, col, 'VD'))
                                actions_ve.add((row, col, 'VB'))
                                if up_condition:
                                    actions_ve.add((row, col, 'VD'))
                            if right_condition:
                                actions_ve.add((row, col, 'VC'))
                                actions_ve.add((row, col, 'VB'))
                                if up_condition:
                                    actions_ve.add((row, col, 'VC'))
                            if up_condition:
                                actions_ve.add((row, col, 'VC'))
                                actions_ve.add((row, col, 'VD'))
                                if left_condition:
                                    actions_ve.add((row, col, 'VD'))
                            actions.extend(actions_ve)
                        
                        if piece == 'VD':
                            actions_vd = set()
                            if left_condition:
                                actions_vd.add((row, col, 'VE'))
                                actions_vd.add((row, col, 'VB'))
                                if down_condition:
                                    actions_vd.add((row, col, 'VE'))
                            if right_condition:
                                actions_vd.add((row, col, 'VC'))
                                actions_vd.add((row, col, 'VB'))
                                if down_condition:
                                    actions_vd.add((row, col, 'VC'))
                            if down_condition:
                                actions_vd.add((row, col, 'VC'))
                                actions_vd.add((row, col, 'VE'))
                                if left_condition:
                                    actions_vd.add((row, col, 'VE'))
                            actions.extend(actions_vd)
                                        
                                                
                    if piece.startswith('F'):
                        
                        left_condition = left in {'FC','FB','FE','BE','VC','LV','VE'}
                        right_condition = right in {'FC','FB','FD','BD','VB','LV','VD'}
                        up_condition = up in {'FC','FE','FD','BC','VC','VD','LH'}
                        down_condition = down in {'FB','FE','FD','BC','VC','VD','LH'}
                        
                        if piece == 'FB':
                            actions_fb = set()
                            if left_condition:
                                actions_fb.add((row, col, 'FD'))
                                actions_fb.add((row, col, 'FC'))
                                if up_condition:
                                    actions_fb.add((row, col, 'FD'))
                            if right_condition:
                                actions_fb.add((row, col, 'FE'))
                                actions_fb.add((row, col, 'FC'))
                                if up_condition:
                                    actions_fb.add((row, col, 'FE'))
                            if up_condition:
                                actions_fb.add((row, col, 'FE'))
                                actions_fb.add((row, col, 'FD'))
                                if left_condition:
                                    actions_fb.add((row, col, 'FD'))
                            actions.extend(actions_fb)
                            
                                    
                        if piece == 'FC':
                            actions_fc = set()
                            if left_condition:
                                actions_fc.add((row, col, 'FD'))
                                actions_fc.add((row, col, 'FB'))
                                if down_condition:
                                    actions_fc.add((row, col, 'FD'))
                            if right_condition:
                                actions_fc.add((row, col, 'FE'))
                                actions_fc.add((row, col, 'FB'))
                                if down_condition:
                                    actions_fc.add((row, col, 'FE'))
                            if down_condition:
                                actions_fc.add((row, col, 'FE'))
                                actions_fc.add((row, col, 'FD'))
                                if left_condition:
                                    actions_fc.add((row, col, 'FD'))
                            actions.extend(actions_fc)
                            
                        if piece == 'FE':
                            actions_fe = set()
                            if left_condition:
                                actions_fe.add((row, col, 'FD'))
                                actions_fe.add((row, col, 'FB'))
                                if up_condition:
                                    actions_fe.add((row, col, 'FD'))
                            if right_condition:
                                actions_fe.add((row, col, 'FC'))
                                actions_fe.add((row, col, 'FB'))
                                if up_condition:
                                    actions_fe.add((row, col, 'FC'))
                            if up_condition:
                                actions_fe.add((row, col, 'FC'))
                                actions_fe.add((row, col, 'FD'))
                                if left_condition:
                                    actions_fe.add((row, col, 'FD'))
                            actions.extend(actions_fe)
                            
                        if piece == 'FD':
                            actions_fd = set()
                            if left_condition:
                                actions_fd.add((row, col, 'FE'))
                                actions_fd.add((row, col, 'FB'))
                                if down_condition:
                                    actions_fd.add((row, col, 'FE'))
                            if right_condition:
                                actions_fd.add((row, col, 'FC'))
                                actions_fd.add((row, col, 'FB'))
                                if down_condition:
                                    actions_fd.add((row, col, 'FC'))
                            if down_condition:
                                actions_fd.add((row, col, 'FC'))
                                actions_fd.add((row, col, 'FE'))
                                if left_condition:
                                    actions_fd.add((row, col, 'FE'))
                            actions.extend(actions_fd)
                            
                    if piece == 'LV':
                        if up in {'FC','FE','FD','BC','VC','VD','LH'} or down in {'FB','FE','FD','BC','VC','VD','LH'}:
                            actions.append((row, col, 'LH'))
                    
                    if piece == 'LH':
                        if left in {'FD','BC','BB','BD','VB','VD','LH'} or right in {'FE','BC','BB','BE','VC','VE','LH'}:
                            actions.append((row, col, 'LV'))
                    
                    
                    

        actions = set(actions)
        #print("actions:",actions)    
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
        """Heuristic for the PipeMania problem: prefer nodes with pieces starting with 'B', then 'V' or 'L', and finally 'F'."""
        state = node.state

    

        return self.expected_connections - state.board.count_real_connections()[0] + self.count_unconnected_pipes(state.board)

    
    def print(self):
        return '\n'.join(' '.join(row) for row in self.board_list)
    


if __name__ == "__main__":
    # Parse the instance
    board = Board.parse_instance()
    # Create a PipeMania object
    pipe_mania = PipeMania(board)
    # Solve the problem using A* search
    solution = greedy_search(pipe_mania, pipe_mania.h)
    # Print the solution
    print(solution.state.board.print())
    


