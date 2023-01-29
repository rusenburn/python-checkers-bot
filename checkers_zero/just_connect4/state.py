import numpy as np
from checkers_zero.state import State


N_COLS = 7
N_ROWS = 6
NO_LAST_ACTION = -1
class JustConnect4State(State):
    def __init__(self,observation:np.ndarray,last_action:int,turn:int) -> None:
        super().__init__()
        self._observation = observation
        self._actions_legality :np.ndarray|None = None
        self._is_terminal : bool|None = None
        self._turn = turn
        self._short :None|tuple= None
        self._last_action = last_action
    
    @property
    def observation_space(self) -> tuple:
        # TODO
        return self._observation.shape
    
    @property
    def n_actions(self)->int:
        return N_COLS
    
    def get_actions_legality(self) -> np.ndarray:
        if self._actions_legality is not None:
            return self._actions_legality.copy()
        actions_legality = np.zeros((N_COLS,),dtype=np.int32)
        for i in range(N_COLS):
            if self._is_legal_action(i):
                actions_legality[i] = 1
        self._actions_legality = actions_legality
        return self._actions_legality.copy()
    
    def is_terminal(self) -> bool:
        if self._is_terminal is not None:
            return self._is_terminal
        if self._last_action == NO_LAST_ACTION:
            return False
        
        player :int = self._observation[2][0][0]
        other:int = 1-player
        if self._is_winning(other) or self._is_winning(player) or self._is_full():
            self._is_game_over = True
        else:
            self._is_game_over = False
        return self._is_game_over
    
    def game_result(self) -> np.ndarray:
        assert self.is_terminal()
        player = self._observation[2][0][0]
        other = 1-player
        wdl = np.zeros((3,),dtype=np.int32)
        if self._is_winning(player):
            wdl[0] = 1
        elif self._is_winning(other):
            wdl[2] = 1
        else:
            wdl[1] = 1
        return wdl.copy()
    
    def step(self, action: int) -> 'JustConnect4State':
        actions_legality = self.get_actions_legality()
        if actions_legality[action] == 0:
            raise ValueError(f"action {action} is not allowed")
        
        player = self._observation[2,0,0]
        previous_player = self._observation[3,0,0]
        other = 1-player
        # change player if this is his 2nd consecutive turn
        next_player = other if previous_player == player else player

        new_obs = self._observation.copy()
        new_obs[2] = next_player
        new_obs[3] = player
        col = action
        row = 0
        while row < N_ROWS:
            if new_obs[player,row,col] == 0 and new_obs[other,row,col] == 0:
                break
            row+=1
            
        new_obs[player,row,col] = 1
        return JustConnect4State(new_obs,action,self._turn+1)


    def render(self) -> None:
        string_list: list[str] = []
        player = self._observation[2][0][0]
        player_rep = ''
        if player == 0:
            player_rep = 'X'
        else:
            player_rep = 'O'
        string_list.append('****************************\n')
        string_list.append(f'*** Player {player_rep} has to move ***\n')
        string_list.append('****************************\n')
        for row in range(6):
            string_list.append('\n')
            string_list.append('____' * N_COLS)
            string_list.append('\n')
            for col in range(N_COLS):
                string_list.append('|')
                if self._observation[0][N_ROWS-row-1][col] == 1:
                    string_list.append(' X ')
                elif self._observation[1][N_ROWS-row-1][col] == 1:
                    string_list.append(' O ')
                else:
                    string_list.append('   ')
                if col == N_COLS-1:  # 0-index last column
                    string_list.append('|')
            if row == N_ROWS-1:  # 0-index last row
                string_list.append('\n')
                string_list.append('----' * N_COLS)
        string_list.append('\n')
        for i in range(N_COLS):
            string_list.append(f'  {i} ')
        print("".join(string_list))

    def to_obs(self) -> np.ndarray:
        return self._observation.copy()
    
    def to_short(self) -> tuple:
        if self._short is not None:
            return self._short
        player = self._observation[2,0,0]
        previous_player = self._observation[3,0,0]
        space: np.ndarray = self._observation[0] - self._observation[1]
        self._short = (player, previous_player,*space.copy().flatten(),)
        return self._short
    
    def get_symmetries(self, probs: np.ndarray) -> list[tuple[State, np.ndarray]]:
        obs_1 = self._observation[:,:,::-1]
        sym_last_action = N_COLS-1-self._last_action
        sym_state_1 = JustConnect4State(obs_1,sym_last_action,self._turn)
        sym_probs_1 = probs[::-1]
        return [(sym_state_1,sym_probs_1)]
    
    def player_turn(self) -> int:
        return self._observation[2,0,0]

    def _is_winning(self,player:int)->bool:
        if self._last_action == NO_LAST_ACTION:
            return False
        
        other = 1 - player
        row = 0
        for row  in range(N_ROWS):
            if row+1 == N_ROWS or (
                self._observation[player][row+1][self._last_action] == 0 and
                self._observation[other][row+1][self._last_action] == 0):
                break
        if self._observation[player,row,self._last_action] != 1:
            return False
        if self._is_vertical_win(player,row,self._last_action):
            return True
        
        if self._is_horizontal_win(player,row,self._last_action):
            return True
        if self._is_forward_diagonal_win(player,row,self._last_action):
            return True
        if self._is_backward_diagonal_win(player,row,self._last_action):
            return True
        return False

    def _is_full(self)->bool:
        game_blocks = N_COLS * N_ROWS
        return self._turn == game_blocks -1 # 0-based
    
    def _is_vertical_win(self,player, row, col)->bool:
        count = 1
        current_col =col+1
        while current_col < N_COLS and self._observation[player][row][current_col]:
            count += 1
            current_col += 1

        current_col = col-1
        while current_col >= 0 and self._observation[player][row][current_col]:
            count += 1
            current_col -= 1
        return count >= 4
    
    def _is_horizontal_win(self,player,row,col)->bool:
        count = 1
        current_row = row+1
        while current_row < N_ROWS and self._observation[player][current_row][col]:
            count += 1
            current_row += 1
        current_row = row-1
        while current_row >= 0 and self._observation[player][current_row][col]:
            count += 1
            current_row -= 1
        return count >= 4


    def _is_forward_diagonal_win(self,player,row,col)->bool:
        count = 1
        i=1
        while row+i < N_ROWS and col+i < N_COLS and self._observation[player][row+i][col+i]:
            count += 1
            i += 1
        i = 1
        while row-i >= 0 and col-i >= 0 and self._observation[player][row-i][col-i]:
            count += 1
            i += 1
        return count >= 4
    
    def _is_backward_diagonal_win(self,player,row,col)->bool:
        count =1
        i = 1
        while row+i < N_ROWS and col-i >= 0 and self._observation[player][row+i][col-i]:
            count += 1
            i += 1
        i = 1
        while row-i >= 0 and col+i < N_COLS and self._observation[player][row-i][col+i]:
            count += 1
            i += 1
        return count >= 4
    
    def _is_legal_action(self,action:int):
        last_col_idx = N_COLS-1
        col= action
        if col > last_col_idx or col < 0:
            return False
        player: int = self._observation[2][0][0]
        other: int = 1-player
        last_row = N_ROWS-1
        # check if the top cell in the col is empty
        result = self._observation[player][last_row][col] == 0 and self._observation[other][last_row][col] == 0
        return result

    

    