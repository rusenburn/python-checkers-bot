import numpy as np
from checkers_zero.state import State

ROWS :int = 8
COLS: int = 8
N_ACTIONS = ROWS*COLS * (ROWS-1 + COLS-1)
EMPTY_CELL = 0
PLAYER_P_CELL = 1
PLAYER_K_CELL = 2
OPPONENT_P_CELL = -1
OPPONENT_K_CELL = -2
DIRECTIONS = (
    [0,-1], # LEFT 
    [0,1], # RIGHT
    [-1,0], # UP
    [1,0] # DOWN
    )


'''
-------------------------------------------------
|  0  | 14  | 28  | 42  | 56  | 70  | 84  | 98  |
-------------------------------------------------
| 112 | 126 | 140 | 154 | 168 | 182 | 196 | 210 |
-------------------------------------------------
| 224 | 238 | 252 | 266 | 280 | 294 | 308 | 322 |
-------------------------------------------------
| 336 | 350 | 364 | 378 | 392 | 406 | 420 | 434 |
-------------------------------------------------
| 448 | 462 | 476 | 490 | 504 | 518 | 532 | 546 |
-------------------------------------------------
| 560 | 574 | 588 | 602 | 616 | 630 | 644 | 658 |
-------------------------------------------------
| 672 | 686 | 700 | 714 | 728 | 742 | 756 | 770 |
-------------------------------------------------
| 784 | 798 | 812 | 826 | 840 | 854 | 868 | 882 |


-------------------------------------------------
|     |     |     | 007 |     |     |     |     |
-------------------------------------------------
|     |     |     | 008 |     |     |     |     |
-------------------------------------------------
|     |     |     | 009 |     |     |     |     |
-------------------------------------------------
| 000 | 001 | 002 |  X  | 003 | 004 | 005 | 006 |
-------------------------------------------------
|     |     |     | 010 |     |     |     |     |
-------------------------------------------------
|     |     |     | 011 |     |     |     |     |
-------------------------------------------------
|     |     |     | 012 |     |     |     |     |
-------------------------------------------------
|     |     |     | 013 |     |     |     |     |
-------------------------------------------------

'''
class TurkishDraughtsState(State):
    def __init__(self,board:np.ndarray,n_no_capture_rounds:int,last_jump:tuple[int,int]|None,last_jump_legal_actions:np.ndarray|None,current_player:int) -> None:
        super().__init__()
        self._board = board
        self._n_no_capture_rounds = n_no_capture_rounds
        self._last_jump = last_jump
        self._last_jump_legal_actions = last_jump_legal_actions
        self._current_player = current_player

        self._cached_actions_legality :np.ndarray | None= None
        self._cached_is_terminal: bool|None = None
        self._cached_game_result : np.ndarray|None = None
        self._cached_observation :np.ndarray|None=  None
        self._cached_short : tuple | None = None
    
    @property
    def n_actions(self) -> int:
        return N_ACTIONS
    
    @property
    def observation_space(self) -> tuple:
        return (6,ROWS,COLS)
    

    def get_actions_legality(self) -> np.ndarray:
        if self._cached_actions_legality is not None:
            return self._cached_actions_legality.copy()
        if self._last_jump_legal_actions is not None:
            assert self._last_jump is not None
            self._cached_actions_legality = self._last_jump_legal_actions
            return self._cached_actions_legality.copy()
        
        assert self._last_jump is None
        no_capture = True
        actions_legality_no_capture = np.zeros((N_ACTIONS),dtype=np.int32)
        actions_legality_capture = np.zeros((N_ACTIONS),dtype=np.int32)
        for row in range(ROWS):
            for col in range(COLS):
                if self._board[row,col] == 0:
                    continue
                
                elif self._board[row,col] == PLAYER_P_CELL:
                    cell_actions_index = (row*COLS+col) * (ROWS+COLS-2)
                    if col-1 >= 0:
                        if self._board[row,col-1] == 0 and no_capture: # LEFT is EMPTY
                            target_index = col-1
                            legal_action = cell_actions_index+target_index
                            actions_legality_no_capture[legal_action] =1
                            
                        elif col-2 >=0 and self._board[row,col-1] <0 and self._board[row,col-2] == 0 :# LEFT has opponent that can be captured
                            target_index = col-2
                            legal_action = cell_actions_index+target_index
                            no_capture = False
                            actions_legality_capture[legal_action]=1
                            
                    if col+1 < COLS:
                        if self._board[row,col+1] == 0 and no_capture: # RIGHT is EMPTY
                            target_index = col + 1 - 1
                            legal_action = cell_actions_index+target_index
                            actions_legality_no_capture[legal_action]=1
                        elif col+2 <COLS and self._board[row,col+1] < 0 and self._board[row,col+2] == 0: # RIGHT is capurable opponent
                            target_index = col + 2 - 1
                            legal_action = cell_actions_index + target_index
                            no_capture = False
                            actions_legality_capture[legal_action]=1
                    if row-1 >=0:
                        if self._board[row-1,col] == 0 and no_capture : # UP is empty
                            target_index = COLS-1 + row-1
                            legal_action = cell_actions_index + target_index
                            actions_legality_no_capture[legal_action] = 1
                        elif row-2>=0 and self._board[row-1,col] < 0 and self._board[row-2,col]==0: # UP is capturable opponent
                            target_index = COLS-1 + row-2
                            legal_action = cell_actions_index + target_index
                            no_capture = False
                            actions_legality_capture[legal_action]=1
                    # if row+1 < ROWS:
                    #     if self._board[row+1,col] == 0 and no_capture: # DOWN is EMPTY
                    #         target_index = COLS-1 + row + 1 - 1
                    #         legal_action = cell_actions_index + target_index
                    #         actions_legality_no_capture[legal_action]=1
                    #     elif row+2 < ROWS and self._board[row+1,col] < 0 and self._board[row+2,col] == 0: # DOWN is capturable opponent
                    #         target_index = COLS -1 + row + 2 - 1
                    #         legal_action = cell_actions_index + target_index
                    #         no_capture = False
                    #         actions_legality_capture[legal_action]=1
                elif self._board[row,col] == PLAYER_K_CELL:
                    cell_actions_index = (row*COLS+col) * (ROWS+COLS-2)
                    for col_dir in range(1,COLS):
                        target_col = col - col_dir
                        if target_col < 0 :
                            break
                        if self._board[row,target_col] == 0:
                            if no_capture:
                                target_index = target_col
                                legal_action = cell_actions_index + target_index
                                actions_legality_no_capture[legal_action] = 1
                        else:
                            if self._board[row,target_col] < 0:
                                second_target = target_col - 1
                                while second_target >=0 and self._board[row,second_target] == 0:
                                    target_index = second_target
                                    legal_action = cell_actions_index + target_index
                                    no_capture = False
                                    actions_legality_capture[legal_action] = 1
                                    second_target -=1
                            break
                    for col_dir in range(1,COLS):
                        target_col = col + col_dir
                        if target_col >= COLS:
                            break
                        if self._board[row,target_col] == 0:
                            if no_capture:
                                target_index = target_col - 1
                                legal_action = cell_actions_index + target_index
                                actions_legality_no_capture[legal_action] = 1
                        else:
                            if self._board[row,target_col] < 0:
                                second_target = target_col + 1
                                while second_target < COLS and self._board[row,second_target] == 0:
                                    target_index = second_target -1
                                    legal_action = cell_actions_index + target_index
                                    no_capture = False
                                    actions_legality_capture[legal_action] = 1
                                    second_target +=1
                            break
                    
                    for row_dir in range(1,ROWS):
                        target_row = row - row_dir
                        if target_row < 0:
                            break
                        if self._board[target_row,col] == 0:
                            if no_capture:
                                target_index = COLS-1 + target_row
                                legal_action = cell_actions_index + target_index
                                actions_legality_no_capture[legal_action] = 1
                        else:
                            if self._board[target_row,col] < 0 :
                                second_target = target_row - 1
                                while second_target >= 0 and self._board[second_target,col] == 0:
                                    target_index = COLS-1 + second_target
                                    legal_action = cell_actions_index + target_index
                                    actions_legality_capture[legal_action] = 1
                                    no_capture = False
                                    second_target -=1
                            break
                    
                    for row_dir in range(1,ROWS):
                        target_row = row + row_dir
                        if target_row >= ROWS:
                            break
                        if self._board[target_row,col] == 0:
                            if no_capture:
                                target_index = COLS-1 + target_row - 1
                                legal_action = cell_actions_index + target_index
                                actions_legality_no_capture[legal_action] = 1
                        else:
                            if self._board[target_row,col] <0:
                                second_target = target_row + 1
                                while second_target < ROWS and self._board[second_target,col]==0:
                                    target_index = COLS-1 + second_target - 1
                                    legal_action = cell_actions_index + target_index
                                    actions_legality_capture[legal_action]=1
                                    no_capture=False
                                    second_target +=1
                            break
        
        if no_capture:
            self._cached_actions_legality = actions_legality_no_capture
        else :
            self._cached_actions_legality = actions_legality_capture
        return self._cached_actions_legality.copy()

    def is_terminal(self) -> bool:
        if self._cached_is_terminal is not None:
            return self._cached_is_terminal
        if self._is_opponent_win():
            self._cached_is_terminal = True
            return self._cached_is_terminal
        if self._is_draw():
            self._cached_is_terminal = True
            return self._cached_is_terminal

        self._cached_is_terminal = False
        return self._cached_is_terminal
    
    def game_result(self) -> np.ndarray:
        assert self.is_terminal()
        if self._cached_game_result is not None:
            return self._cached_game_result.copy()
        wdl = np.zeros((3,),dtype=np.int32)
        if self._is_opponent_win():
            wdl[2] = 1
        elif self._is_draw():
            wdl[1] = 1
        else:
            wdl[0] = 1
            raise AssertionError("unreachable")
        
        self._cached_game_result = wdl
        return self._cached_game_result.copy()
    
    def step(self, action: int) -> 'TurkishDraughtsState':
        if self.get_actions_legality()[action]!=1:
            raise ValueError(f"actions {action} is not a legal action.")
        row,col,target_row,target_col = self._decode_action(action)
        new_board = self._board.copy()
        assert new_board[row,col] > 0

        # move current piece
        new_board[target_row,target_col] = new_board[row,col]
        # if promotion
        if target_row == ROWS-1 or target_row == 0:
            new_board[target_row,target_col] = PLAYER_K_CELL
        new_board[row,col] = 0
        row_dir = 0 if target_row - row == 0 else (target_row-row) // abs(target_row-row)
        col_dir = 0 if target_col - col == 0 else (target_col-col) // abs(target_col-col)

        current_row , current_col = row + row_dir , col + col_dir
        capture = False
        while current_row != target_row or current_col != target_col:
            if new_board[current_row,current_col] !=0:
                capture = True
                assert new_board[current_row,current_col] < 0
            new_board[current_row,current_col] = 0
            current_col+= col_dir
            current_row+=row_dir
        if capture:
            capture_actions_legality = np.zeros((N_ACTIONS,),dtype=np.int32)
            no_capture_rounds = 0
            self._assign_legal_action(
                board=new_board,
                row=target_row,
                col=target_col,
                capture_only=True,
                out_actions_legality_no_capture=np.zeros(1,),
                out_actions_legality_capture=capture_actions_legality)
        
            if capture_actions_legality.max() > 0:
                next_player = self._current_player
                last_jump = (target_row,target_col)
                last_jump_actions_legality = capture_actions_legality
                
            else:
                next_player = 1-self._current_player
                last_jump = None
                last_jump_actions_legality = None
                new_board = new_board[::-1,::-1]
                new_board*=-1

        else:
            next_player = 1-self._current_player
            last_jump = None
            last_jump_actions_legality = None
            no_capture_rounds = self._n_no_capture_rounds+1
            new_board = new_board[::-1,::-1]
            new_board*=-1
        
        return TurkishDraughtsState(
                board=new_board,
                n_no_capture_rounds=no_capture_rounds,
                last_jump=last_jump,
                last_jump_legal_actions=last_jump_actions_legality,
                current_player=next_player)
    
    def to_obs(self) -> np.ndarray:
        if self._cached_observation is not None:
            return self._cached_observation.copy()
        
        
        obs = np.zeros((6,ROWS,COLS),dtype=np.int32)
        # BOARD OBSERVATION
        obs[0,:,:] = self._board == PLAYER_P_CELL
        obs[1,:,:] = self._board == PLAYER_K_CELL
        obs[2,:,:] = self._board == OPPONENT_P_CELL
        obs[3,:,:] = self._board == OPPONENT_K_CELL

        # LAST_JUMP OBSERVATION
        if self._last_jump is not None:
            row,col = self._last_jump
            obs[4,row,col] = 1
        
        # Number of no capture consecutive rounds observation
        row = self._n_no_capture_rounds // COLS
        col = self._n_no_capture_rounds % COLS
        obs[5,row,col] = 1

        # cache observation
        self._cached_observation = obs

        return self._cached_observation.copy()

    
    def to_short(self) -> tuple:
        if self._cached_short is not None:
            return self._cached_short
        self._cached_short = (self._current_player,*self._board.flatten(),self._n_no_capture_rounds,self._last_jump)
        return self._cached_short
    
    def player_turn(self) -> int:
        return self._current_player
    
    def render(self) -> None:
        return super().render()
    
    def get_symmetries(self, probs: np.ndarray) -> list[tuple['State', np.ndarray]]:
        return []
    
    @staticmethod
    def _assign_legal_action(board:np.ndarray,row:int,col:int,capture_only:bool,out_actions_legality_no_capture:np.ndarray,out_actions_legality_capture:np.ndarray)->None:
        no_capture = not capture_only
        if board[row,col] == 0:
            return 
        elif board[row,col] == PLAYER_P_CELL:
            cell_actions_index = (row*COLS+col) * (ROWS+COLS-2)
            if col-1 >= 0:
                if board[row,col-1] == 0 and no_capture: # LEFT is EMPTY
                    target_index = col-1
                    legal_action = cell_actions_index+target_index
                    out_actions_legality_no_capture[legal_action] =1
                    
                elif col-2 >=0 and board[row,col-1] <0 and board[row,col-2] == 0 :# LEFT has opponent that can be captured
                    target_index = col-2
                    legal_action = cell_actions_index+target_index
                    no_capture = False
                    out_actions_legality_capture[legal_action]=1
                    
            if col+1 < COLS:
                if board[row,col+1] == 0 and no_capture: # RIGHT is EMPTY
                    target_index = col + 1 - 1
                    legal_action = cell_actions_index+target_index
                    out_actions_legality_no_capture[legal_action]=1
                elif col+2 <COLS and board[row,col+1] < 0 and board[row,col+2] == 0: # RIGHT is capurable opponent
                    target_index = col + 2 - 1
                    legal_action = cell_actions_index + target_index
                    no_capture = False
                    out_actions_legality_capture[legal_action]=1
            if row-1 >=0:
                if board[row-1,col] == 0 and no_capture : # UP is empty
                    target_index = COLS-1 + row-1
                    legal_action = cell_actions_index + target_index
                    out_actions_legality_no_capture[legal_action] = 1
                elif row-2>=0 and board[row-1,col] < 0 and board[row-2,col]==0: # UP is capturable opponent
                    target_index = COLS-1 + row-2
                    legal_action = cell_actions_index + target_index
                    no_capture = False
                    out_actions_legality_capture[legal_action]=1
            # if row+1 < ROWS:
            #     if board[row+1,col] == 0 and no_capture: # DOWN is EMPTY
            #         target_index = COLS-1 + row + 1 - 1
            #         legal_action = cell_actions_index + target_index
            #         out_actions_legality_no_capture[legal_action]=1
            #     elif row+2 < ROWS and board[row+1,col] < 0 and board[row+2,col] == 0: # DOWN is capturable opponent
            #         target_index = COLS -1 + row + 2 - 1
            #         legal_action = cell_actions_index + target_index
            #         no_capture = False
            #         out_actions_legality_capture[legal_action]=1
        elif board[row,col] == PLAYER_K_CELL:
            cell_actions_index = (row*COLS+col) * (ROWS+COLS-2)
            for col_dir in range(1,COLS):
                target_col = col - col_dir
                if target_col < 0 :
                    break
                if board[row,target_col] == 0:
                    if no_capture:
                        target_index = target_col
                        legal_action = cell_actions_index + target_index
                        out_actions_legality_no_capture[legal_action] = 1
                else:
                    if board[row,target_col] < 0:
                        second_target = target_col - 1
                        while second_target >=0 and board[row,second_target] == 0:
                            target_index = second_target
                            legal_action = cell_actions_index + target_index
                            no_capture = False
                            out_actions_legality_capture[legal_action] = 1
                            second_target -=1
                    break
            for col_dir in range(1,COLS):
                target_col = col + col_dir
                if target_col >= COLS:
                    break
                if board[row,target_col] == 0:
                    if no_capture:
                        target_index = target_col - 1
                        legal_action = cell_actions_index + target_index
                        out_actions_legality_no_capture[legal_action] = 1
                else:
                    if board[row,target_col] < 0:
                        second_target = target_col + 1
                        while second_target < COLS and board[row,second_target] == 0:
                            target_index = second_target -1
                            legal_action = cell_actions_index + target_index
                            no_capture = False
                            out_actions_legality_capture[legal_action] = 1
                            second_target +=1
                    break
            
            for row_dir in range(1,ROWS):
                target_row = row - row_dir
                if target_row < 0:
                    break
                if board[target_row,col] == 0:
                    if no_capture:
                        target_index = COLS-1 + target_row
                        legal_action = cell_actions_index + target_index
                        out_actions_legality_no_capture[legal_action] = 1
                else:
                    if board[target_row,col] < 0 :
                        second_target = target_row - 1
                        while second_target >= 0 and board[second_target,col] == 0:
                            target_index = COLS-1 + second_target
                            legal_action = cell_actions_index + target_index
                            out_actions_legality_capture[legal_action] = 1
                            no_capture = False
                            second_target -=1
                    break
            
            for row_dir in range(1,ROWS):
                target_row = row + row_dir
                if target_row >= ROWS:
                    break
                if board[target_row,col] == 0:
                    if no_capture:
                        target_index = COLS-1 + target_row - 1
                        legal_action = cell_actions_index + target_index
                        out_actions_legality_no_capture[legal_action] = 1
                else:
                    if board[target_row,col] <0:
                        second_target = target_row + 1
                        while second_target < ROWS and board[second_target,col]==0:
                            target_index = COLS-1 + second_target - 1
                            legal_action = cell_actions_index + target_index
                            out_actions_legality_capture[legal_action]=1
                            no_capture=False
                            second_target +=1
                    break
    
    def _is_opponent_win(self):
        have_pieces = np.any(self._board>0)
        if have_pieces is False:
            return True
        actions_legality = self.get_actions_legality()
        if actions_legality.sum() == 0:
            return True
        return False
    
    def _is_draw(self):
        return self._n_no_capture_rounds == 40
    @staticmethod
    def _decode_action(action:int)->tuple[int,int,int,int]:
        target_index = action % (ROWS+COLS-2) # each specific cell has move upto width + height - 2 (for self) ( 14 normally if cols=8 and rows=8)
        cell_index = action // (ROWS+COLS-2)
        row = cell_index // COLS
        col = cell_index % COLS
        if target_index < COLS-1: # left right move
            target_row = row
            if target_index >= col:
                target_col = target_index+1
            else:
                target_col = target_index
        else: # updown move
            target_index -= (COLS-1)
            target_col = col
            if target_index >= row:
                target_row = target_index+1
            else:
                target_row = target_index
        return row,col,target_row,target_col
    
    @staticmethod
    def _encode_action(row:int,col:int,target_row:int,target_col:int)->int:
        if  row != target_row and col != target_col:
            raise ValueError("Action cannot be encoded")
        cell_actions_index = (row*COLS+col) * (ROWS+COLS-2)
        if target_row != row:
            if target_row > row:
                target_index = target_row-1
            else:
                target_index = target_row
            target_index += COLS-1
        else: # target_col != col
            if target_col > col:
                target_index = target_col-1
            else:
                target_index = target_col
        
        action = cell_actions_index + target_index
        return action
    
    @staticmethod
    def _test_action_encoding():
        for expected_action in range(N_ACTIONS):
            row,col, target_row,target_col=TurkishDraughtsState._decode_action(expected_action)
            actual_action = TurkishDraughtsState._encode_action(row,col,target_row,target_col)
            # print(f"Expected : {expected_action} , Actual: {actual_action}")
            if expected_action != actual_action:
                raise AssertionError(f"Expected : {expected_action} , Actual: {actual_action}")
    
    @staticmethod
    def initialize_new_state(all_kings_mode:bool)->'TurkishDraughtsState':
        board = np.array([
            [0,0,0,0,0,0,0,0],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0]],dtype=np.int32)
        if all_kings_mode:
            board *=2
        state = TurkishDraughtsState(
                board=board,
                n_no_capture_rounds=0,
                last_jump=None,
                last_jump_legal_actions=None,
                current_player=0)
        return state




        


                        
                        
                        



    
    

