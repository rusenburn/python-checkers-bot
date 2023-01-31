import numpy as np
from checkers_zero.state import State

class EnglishDraughtsState(State):
    ROWS: int = 8
    COLS: int = 8
    PLAYER_P_CHANNEL = 0
    PLAYER_K_CHANNEL = 1
    OPPONENT_P_CHANNEL = 2
    OPPONENT_K_CHANNEL = 3
    N_ACTIONS = ROWS*COLS//2*4
    # [upright,upleft,downright,downleft]
    DIRECTIONS = [(-1, 1), (-1, -1), (1, 1), (1, -1)]

    def __init__(self, observation: np.ndarray, n_no_capture_rounds: int, no_kill_shorts: dict, last_jump: tuple[int, int] | None, last_jump_legal_actions: np.ndarray | None, current_player: int) -> None:
        super().__init__()
        self.observation = observation
        self.n_no_capture_rounds = n_no_capture_rounds
        self.no_kill_shorts = no_kill_shorts
        self.last_jump_legal_actions = last_jump_legal_actions
        self.last_jump = last_jump
        self.current_player = current_player

        # cache
        self._cached_actions_legality :np.ndarray|None=None
        self._cached_is_terminal:bool|None = None
        self._cached_game_result:np.ndarray|None = None
        self._cached_observation:np.ndarray|None = None

    @property
    def n_actions(self) -> int:
        return self.N_ACTIONS

    @property
    def observation_space(self) -> tuple:
        return (6,8,8)

    def get_actions_legality(self) -> np.ndarray:
        if self._cached_actions_legality is not None:
            self._cached_actions_legality.copy()
        if self.last_jump_legal_actions is not None:
            assert self.last_jump is not None
            self._cached_actions_legality = self.last_jump_legal_actions
            return self._cached_actions_legality.copy()

        assert self.last_jump is None
        actions_legality_no_capture = np.zeros((self.N_ACTIONS))
        actions_legality_capture = np.zeros((self.N_ACTIONS))
        return_capture = False
        for i in range(self.N_ACTIONS):
            legal, capture = self._is_legal_action(self.observation, i)
            if legal:
                actions_legality_no_capture[i] = 1
            if capture:
                actions_legality_capture[i] = 1
                return_capture = True
        if return_capture:
            self._cached_actions_legality = actions_legality_capture
            return self._cached_actions_legality.copy()
        else:
            self._cached_actions_legality = actions_legality_no_capture
            return self._cached_actions_legality.copy()

    def is_terminal(self) -> bool:
        if self._cached_is_terminal is not None:
            self._cached_is_terminal
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
            return self._cached_game_result
        wdl = np.zeros((3,), dtype=np.int32)
        if self._is_opponent_win():
            wdl[2] = 1
        elif self._is_draw():
            wdl[1] = 1
        else:
            wdl[0] = 1
            raise AssertionError("unreachable")
        self._cached_game_result = wdl
        return self._cached_game_result.copy()

    def step(self, action: int) -> 'EnglishDraughtsState':
        if self.get_actions_legality()[action] != 1:
            raise ValueError(f"actions {action} is not a legal action.")
        row, col, direction = self._get_row_col_direction_from_action(action)
        row_dir, col_dir = direction
        target_row, target_col = row+row_dir, col+col_dir
        n_no_capture_rounds = self.n_no_capture_rounds
        new_obs = self.observation.copy()
        is_moving_piece_king = self._is_king(new_obs, row, col)
        our_channel = self.PLAYER_K_CHANNEL if is_moving_piece_king else self.PLAYER_P_CHANNEL
        switch_player = True
        next_player = 1-self.current_player
        last_jump: tuple[int, int] | None = None
        jump_legal_actions: np.ndarray | None = None
        if self._is_empty_position(self.observation, target_row, target_col):
            # remove the piece
            new_obs[our_channel, row, col] = 0
            # add the piece # check if gonna be promoted
            if target_row == 0 or target_row == self.ROWS-1:
                new_obs[self.PLAYER_K_CHANNEL,target_row,target_col] = 1
            else:
                new_obs[our_channel, target_row, target_col] = 1
            n_no_capture_rounds += 1
        else:  # not empty then we have an opponent , capture and jump
            is_captured_piece_king = new_obs[self.OPPONENT_K_CHANNEL,target_row,target_col] == 1

            opponent_channel = self.OPPONENT_K_CHANNEL if is_captured_piece_king else self.OPPONENT_P_CHANNEL

            # remove our piece
            new_obs[our_channel, row, col] = 0
            # remove opponent piece
            assert new_obs[opponent_channel, target_row, target_col] == 1
            new_obs[opponent_channel, target_row, target_col] = 0

            # add our piece after capturing opponent and check for promotion
            jumpto_row, jumpto_col = target_row+row_dir, target_col + col_dir
            assert self._is_empty_position(
                self.observation, jumpto_row, jumpto_col)
            if jumpto_row == 0 or jumpto_row == self.ROWS-1:
                new_obs[self.PLAYER_K_CHANNEL, jumpto_row, jumpto_col] = 1
            else:
                new_obs[our_channel, jumpto_row, jumpto_col] = 1
            n_no_capture_rounds = 0

            # check if it can double jump [ DO NOT DO DOUBLE JUMP JUST CHECK IT IS POSSIBLE ]

            jump_legal_actions = np.zeros((self.N_ACTIONS,), dtype=np.int32)
            has_another_jump_action = False
            for d in range(4):
                # jumpto is our new starting position in double jump
                jump_action = self._encode_action(jumpto_row, jumpto_col, d)
                _, capture = self._is_legal_action(new_obs, jump_action)
                # it must be a capture
                if capture:
                    jump_legal_actions[jump_action] = 1
                    has_another_jump_action = True
                    next_player = self.current_player
                    switch_player = False
                    last_jump = (jumpto_row,jumpto_col)

            if not has_another_jump_action:
                jump_legal_actions = None
                last_jump = None

        if switch_player:
            # change observation based on current player
            new_obs = new_obs[[self.OPPONENT_P_CHANNEL, self.OPPONENT_K_CHANNEL,
                               self.PLAYER_P_CHANNEL, self.PLAYER_K_CHANNEL], ::-1, ::-1]

        new_state = EnglishDraughtsState(new_obs, n_no_capture_rounds=n_no_capture_rounds, last_jump=last_jump,
                                         no_kill_shorts=dict(), last_jump_legal_actions=jump_legal_actions, current_player=next_player)
        return new_state

    def to_obs(self) -> np.ndarray:
        if self._cached_observation is not None:
            return self._cached_observation.copy()
        no_capture_rounds = self._get_no_capture_rounds_observation()
        last_jump_obs = self._get_last_jump_observation()
        obs = np.concatenate(
            [self.observation[0:4], no_capture_rounds, last_jump_obs], axis=0)
        assert obs.shape == self.observation_space
        self._cached_observation =  obs
        return self._cached_observation.copy()
        # return self.observation.copy()

    def player_turn(self) -> int:
        return self.current_player

    def render(self) -> None:
        raise NotImplementedError()


    def to_short(self) -> tuple:
        short = self.observation[0:4].astype(np.int0)
        return (self.current_player, short.tobytes(), self.n_no_capture_rounds, self.last_jump)

    def get_symmetries(self, probs: np.ndarray) -> list[tuple['EnglishDraughtsState', np.ndarray]]:
        return []

    def _is_opponent_win(self) -> bool:
        if np.sum(self.observation[[self.PLAYER_K_CHANNEL, self.PLAYER_P_CHANNEL]]) == 0:
            return True
        actions_legality = self.get_actions_legality()
        if actions_legality.sum() == 0:
            return True
        return False

    def _is_legal_action(self, observation: np.ndarray, action: int) -> tuple[bool, bool]:
        row, col, direction = self._get_row_col_direction_from_action(action)
        if observation[self.PLAYER_P_CHANNEL][row, col] != 1 and observation[self.PLAYER_K_CHANNEL][row, col] != 1:
            # we do not have a piece here
            return False, False
        row_dir, col_dir = direction
        if not self._is_king(observation, row, col) and self._is_backward_move(direction):
            return False, False
        target_row, target_col = row+row_dir, col+col_dir

        if target_row < 0 or target_row >= self.ROWS or target_col < 0 or target_col >= self.COLS:
            return False, False

        if self._is_empty_position(observation, target_row, target_col):
            return True, False

        if self._can_capture(observation, row, col, direction):
            return True, True
        return False, False

    def _can_capture(self, observation: np.ndarray, row: int, col: int, direction: tuple[int, int]) -> bool:
        row_dir, col_dir = direction
        capture_row, capture_col = row+row_dir, col+col_dir
        jump_row, jump_col = capture_row + row_dir, capture_col + col_dir

        # if inside the board
        if jump_row < 0 or jump_row >= self.ROWS or jump_col < 0 or jump_col >= self.COLS:
            return False

        # if there is an opponent pawn or king in capture cell
        if observation[self.OPPONENT_P_CHANNEL, capture_row, capture_col] == 1 or observation[self.OPPONENT_K_CHANNEL, capture_row, capture_col] == 1:
            if self._is_empty_position(observation, jump_row, jump_col):
                return True
            else:
                return False
        return False

    def _get_row_col_direction_from_action(self, action: int) -> tuple[int, int, tuple[int, int]]:
        a = action // 4
        row = a // (self.COLS//2)
        col = (a % (self.COLS//2)) * 2 + (1-(row % 2))
        d = action % 4
        direction = self.DIRECTIONS[d]
        return row, col, direction

    def _encode_action(self, row: int, col: int, d: int) -> int:
        a = row * (self.COLS//2) + col//2
        action = a * 4 + d
        return action

    def _is_empty_position(self, observation: np.ndarray, row: int, col: int) -> bool:
        return np.sum(observation[0:4, row, col] != 0) == 0

    def _is_king(self, observation: np.ndarray, row: int, col: int) -> bool:
        return observation[self.PLAYER_K_CHANNEL, row, col] == 1

    def _is_backward_move(self, direction: tuple[int, int]) -> bool:
        return direction[0] == 1

    def _is_draw(self):
        return self.n_no_capture_rounds == 40

    def _get_no_capture_rounds_observation(self) -> np.ndarray:
        res = np.zeros((1, self.ROWS, self.COLS), dtype=np.int32)
        row = self.n_no_capture_rounds // self.COLS
        col = self.n_no_capture_rounds % self.COLS
        res[0, row, col] = 1
        return res

    def _get_last_jump_observation(self) -> np.ndarray:
        res = np.zeros((1, self.ROWS, self.COLS), dtype=np.int32)
        if self.last_jump is not None:
            row, col = self.last_jump
            res[0, row, col] = 1
        return res