from sys import stdin
    
class PipeManiaState:
    
    state_id = 0
    
    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1
        
    def __lt__(self, other):
        return self.id < other.id
    

class Board:
    
    def __init__(self, board):
        self.board = board
        
    def adjacent_vertical_values(self, row:int, col:int): 
        above = self.board[row-1][col] if row > 0 else None
        below = self.board[row+1][col] if row+1 < len(self.board) and col < len(self.board[row+1]) else None
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
    
    def get_value(self, row:int, col:int):
        if row >= len(self.board) or col >= len(self.board[row]):
            return None  # ou algum outro valor que indica um erro
        return self.board[row][col]

    def print(self):
        for row in self.board:
            print('\t'.join(row))
            
            
board = Board.parse_instance()

print(board.adjacent_vertical_values(0, 0))
print(board.adjacent_horizontal_values(0, 0))
print(board.adjacent_vertical_values(1, 1))
print(board.adjacent_horizontal_values(1, 1))


