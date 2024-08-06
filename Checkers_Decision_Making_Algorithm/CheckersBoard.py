import numpy as np


'''
Chosen Values
=============
Minimize
Dark piece:
Man = -1
King = -2

Maximize
Light piece:
Man = 1
King = 2

Empty cell = 0
'''

ALPHA_BETA_ENABLE = True

class CheckersBoard:
    def __init__(self):
        """
        Initialise Checkers board
        :param None
            0    ------ c ----->    7
        0   -------------------------
            |       Light side      |
        |   |                       |
        r   |                       |
        |   |                       |
        ↓   |                       |
            |       Dark side       |
        7   -------------------------
        """

        self.num_rows = 8
        self.num_columns = 8

        # list of row and column locations of pieces
        self.rows = np.arange(0, self.num_rows, 1)
        self.columns = np.array([np.arange(1, self.num_columns, 2), np.arange(0, self.num_columns, 2)])

        self.board = np.array([
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0]
        ])

    def reset_board(self):
        self.board = np.array([
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0]
        ])

    def print_board_layout(self, board: np.ndarray):
        horizontal_line_border = "  #########################################"
        horizontal_line_normal = "  #---------------------------------------#"
        vertical_line_border = "#"
        vertical_line_normal = "|"
        space = " "
        column_numbers = "   "
        for c in range(self.num_columns):
            column_numbers += space + str(c) + " " + space + " "

        print(column_numbers)
        print(horizontal_line_border)

        for r in range(self.num_rows):
            new_column_print = str(r) + " " + vertical_line_border + space
            for c in range(self.num_columns):

                if board[r][c] == 0:
                    # no piece in that location
                    new_column_print += "  "
                else:
                    if board[r][c] < 0:
                        # dark
                        new_column_print += "B"
                    else:
                        # light
                        new_column_print += "W"

                    if board[r][c] % 2 == 0:
                        new_column_print += "K"
                    else:
                        new_column_print += " "

                new_column_print += space

                if c < (self.num_columns - 1):
                    new_column_print += vertical_line_normal + space

            new_column_print += vertical_line_border
            print(new_column_print)
            if r < (self.num_rows - 1):
                print(horizontal_line_normal)

        print(horizontal_line_border)

    def get_piece_at(self, r: int, c: int, board: np.ndarray) -> tuple:
        """
        Check for a valid piece in a position
        :param r:
        :param c:
        :param board:
        :return: Within grid flag, piece type
        """
        if (r >= 0) & (r < self.num_rows) & (c >= 0) & (c < self.num_columns):
            return True, board[r][c]
        else:
            return False, 0

    def find_longest_move_list(self, list_of_moves: list) -> list:
        max_length = 0
        max_index = []
        for i in range(len(list_of_moves)):
            if len(list_of_moves[i]) == max_length:
                max_index.append(i)
            elif len(list_of_moves[i]) > max_length:
                max_length = len(list_of_moves[i])
                max_index = [i]

        list_return = []
        for i in max_index:
            # check if the list already exists as one of the longest lists
            if list_of_moves[i] not in list_return:
                list_return.append(list_of_moves[i])

        # return a list with no move
        if len(list_of_moves) == 0:
            list_return.append([])

        return list_return

    def calculate_force_jumps(self, r: int, c: int, board: np.ndarray) -> list:
        """
        get current location of piece to look for jumps
        :param r:
        :param c:
        :param board:
        :return: single or multiple lists of possible moves
        """

        possible_moves = []
        p_current = board[r][c]

        # if light piece, or dark king piece
        if (board[r][c] > 0) | ((board[r][c] < 0) & (board[r][c] % 2 == 0)):
            # down left, jump over opposite colour and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r + 1, c - 1, board)
            valid_dest, p_dest = self.get_piece_at(r + 2, c - 2, board)
            if valid_jump & valid_dest & (p_jump * p_current < 0) & (p_dest == 0):
                # update board to new moved layout
                temp_board = board.copy()
                temp_board[r + 1][c - 1] = 0
                temp_board[r + 2][c - 2] = temp_board[r][c]
                temp_board[r][c] = 0

                moves = self.find_longest_move_list(self.calculate_force_jumps(r + 2, c - 2, temp_board))

                for m in moves:
                    m.insert(0, [r, c])
                    possible_moves.append(m)

            # down right, jump over opposite colour and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r + 1, c + 1, board)
            valid_dest, p_dest = self.get_piece_at(r + 2, c + 2, board)
            if valid_jump & valid_dest & (p_jump * p_current < 0) & (p_dest == 0):
                # update board to new moved layout
                temp_board = board.copy()
                temp_board[r + 1][c + 1] = 0
                temp_board[r + 2][c + 2] = temp_board[r][c]
                temp_board[r][c] = 0

                moves = self.find_longest_move_list(self.calculate_force_jumps(r + 2, c + 2, temp_board))

                for m in moves:
                    m.insert(0, [r, c])
                    possible_moves.append(m)

        # if dark piece, or light king piece
        if (board[r][c] < 0) | ((board[r][c] > 0) & (board[r][c] % 2 == 0)):
            # up left, jump over opposite colour and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r - 1, c - 1, board)
            valid_dest, p_dest = self.get_piece_at(r - 2, c - 2, board)
            if valid_jump & valid_dest & (p_jump * p_current < 0) & (p_dest == 0):
                # update board to new moved layout
                temp_board = board.copy()
                temp_board[r - 1][c - 1] = 0
                temp_board[r - 2][c - 2] = temp_board[r][c]
                temp_board[r][c] = 0

                moves = self.find_longest_move_list(self.calculate_force_jumps(r - 2, c - 2, temp_board))

                for m in moves:
                    m.insert(0, [r, c])
                    possible_moves.append(m)

            # up right, jump over opposite colour and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r - 1, c + 1, board)
            valid_dest, p_dest = self.get_piece_at(r - 2, c + 2, board)
            if valid_jump & valid_dest & (p_jump * p_current < 0) & (p_dest == 0):
                # update board to new moved layout
                temp_board = board.copy()
                temp_board[r - 1][c + 1] = 0
                temp_board[r - 2][c + 2] = temp_board[r][c]
                temp_board[r][c] = 0

                moves = self.find_longest_move_list(self.calculate_force_jumps(r - 2, c + 2, temp_board))

                for m in moves:
                    m.insert(0, [r, c])
                    possible_moves.append(m)

        longest_moves = self.find_longest_move_list(possible_moves)

        if not longest_moves[0]:
            longest_moves[0] = [[r, c]]

        return longest_moves

    def check_force_jump_moves(self, light_side: bool, board: np.ndarray) -> list:
        all_force_jumps = []
        for r in self.rows:
            for c in self.columns[r % 2]:
                p = board[r][c]
                if (light_side & (p > 0)) | (~light_side & (p < 0)):
                    force_jump = self.calculate_force_jumps(r, c, board)
                    if len(force_jump[0]) > 1:
                        for j in force_jump:
                            all_force_jumps.append(j)

        return all_force_jumps

    def calculate_normal_move(self, r: int, c: int, board: np.ndarray) -> list:
        possible_moves = []

        # if light piece, or dark king piece
        if (board[r][c] > 0) | ((board[r][c] < 0) & (board[r][c] % 2 == 0)):
            # down left, normal jump and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r + 1, c - 1, board)
            if valid_jump & (p_jump == 0):
                possible_moves.append([[r, c], [r + 1, c - 1]])

            # down right, normal jump and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r + 1, c + 1, board)
            if valid_jump & (p_jump == 0):
                possible_moves.append([[r, c], [r + 1, c + 1]])

        # if dark piece, or light king piece
        if (board[r][c] < 0) | ((board[r][c] > 0) & (board[r][c] % 2 == 0)):
            # up left, normal jump and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r - 1, c - 1, board)
            if valid_jump & (p_jump == 0):
                possible_moves.append([[r, c], [r - 1, c - 1]])

            # up right, normal jump and open spot at the end
            valid_jump, p_jump = self.get_piece_at(r - 1, c + 1, board)
            if valid_jump & (p_jump == 0):
                possible_moves.append([[r, c], [r - 1, c + 1]])

        return possible_moves

    def calculate_side_possible_moves(self, light_side: bool, board: np.ndarray) -> list:
        """
        :param light_side: side to calculate possible moves
        :param board:
        :return:
        """
        # print("looking for", "light" if light_side else "dark", "side possible moves")

        # get any force jumps for pieces
        force_jumps = self.check_force_jump_moves(light_side, board)

        if len(force_jumps) > 0:
            # print("YES force jumps must be taken")
            return force_jumps
        else:
            # print("No force jumps, looking for normal moves")

            all_possible_moves = []
            for r in self.rows:
                for c in self.columns[r % 2]:
                    p = board[r][c]
                    if (light_side & (p > 0)) | (~light_side & (p < 0)):
                        possible_moves = self.calculate_normal_move(r, c, board)
                        if len(possible_moves) > 0:
                            for p in possible_moves:
                                all_possible_moves.append(p)

            return all_possible_moves

    def score_board(self, board: np.ndarray) -> int:
        total = 0
        for r in self.rows:
            for c in self.columns[r % 2]:
                total += board[r][c]

        return total

    def execute_move(self, moves, board: np.ndarray) -> tuple:
        # Execution move scoring
        # if normal move = 0 extra points
        # if jump move = (jump man) 10, (jump king) 30 [for every jump add values]
        score_jump_man = 10
        score_jump_king = 30
        # if men promotion to king = 15 points
        score_promote = 15

        total_move_execution_score = 0

        # print("Doing move ", moves)

        # save moved piece type/number
        start_piece = board[moves[0][0]][moves[0][1]]
        for i in range(len(moves)):
            # delete moved piece starting position from board
            board[moves[i][0]][moves[i][1]] = 0

            # check for a jump move
            if (i > 0) & ((moves[i - 1][0] - moves[i][0]) % 2 == 0):
                # getting grid location of jumped piece
                r = (moves[i][0] + moves[i - 1][0]) >> 1
                c = (moves[i][1] + moves[i - 1][1]) >> 1

                # test if jump over king, and add jump move execution
                if abs(board[r][c]) % 2 == 0:
                    total_move_execution_score += score_jump_king
                else:
                    total_move_execution_score += score_jump_man

                board[r][c] = 0

        # moving piece to last grid location
        if (((moves[-1][0] == 0) & (start_piece < 0)) | ((moves[-1][0] == 7) & (start_piece > 0))) & (start_piece % 2 == 1):
            # king promotion of man
            # print("King Promotion")
            board[moves[-1][0]][moves[-1][1]] = start_piece * 2
            total_move_execution_score += score_promote
        else:
            board[moves[-1][0]][moves[-1][1]] = start_piece

        # invert score to minimize the dark side score
        if start_piece < 0:
            total_move_execution_score *= -1

        return board, total_move_execution_score

    def print_moves(self, moves: list):
        if len(moves) > 0:
            option_count = 0
            # go through each possible moves
            for m in moves:
                print("Option ", option_count)
                print(m)
                print("=======================")

                option_count += 1
            print("############################")

    def check_game_end(self, board: np.ndarray):
        see_dark = False
        see_light = False

        for r in self.rows:
            for c in self.columns[r % 2]:
                if board[r][c] > 0:
                    see_light = True
                elif board[r][c] < 0:
                    see_dark = True

                # if both colours are still present, game has not ended
                if see_dark & see_light:
                    return False

        # just seeing one colour of no colour sides, game has ended
        return True

    def get_best_move_alpha_beta_search(self, max_depth: int, max_player_turn: bool) -> tuple:
        # calculate the possible moves from the current game state
        possible_moves = self.calculate_side_possible_moves(max_player_turn, self.board.copy())

        # test if just the board needs to be scores
        if (max_depth == 0) | (len(possible_moves) == 0):
            # if no moves are available or looking at the state only (depth = 0)
            ''' can add big value to show a win or lose '''
            return self.score_board(self.board), []
        else:
            move_scores = []
            if max_player_turn:
                for m in possible_moves:
                    move_scores.append(self.get_min_scoring_move(self.board.copy(), m, max_depth - 1, -np.inf, np.inf, 0))
                best_possible_score = max(move_scores)
            else:
                for m in possible_moves:
                    move_scores.append(self.get_max_scoring_move(self.board.copy(), m, max_depth - 1, -np.inf, np.inf, 0))
                best_possible_score = min(move_scores)

            # save the best moves in the selection list
            best_moves = []
            for i in range(len(move_scores)):
                if move_scores[i] == best_possible_score:
                    best_moves.append(possible_moves[i])

            # select one of best moves at random
            selected_move = np.random.randint(0, len(best_moves))

            # OR select the first equally best move
            # selected_move = 0

            # print(saved_scores, saved_moves)

            # only returning the best move
            return best_possible_score, best_moves[selected_move]

    def get_max_scoring_move(self, board: np.ndarray, move: list, depth: int, alpha: int, beta: int, move_execution_score: int) -> int:
        new_state, new_move_score = self.execute_move(move, board.copy())

        if depth == 0:
            # only evaluate state
            return self.score_board(new_state) + new_move_score
        else:
            v = -np.inf
            possible_moves = self.calculate_side_possible_moves(True, new_state.copy())

            for m in possible_moves:
                v = max(self.get_min_scoring_move(new_state.copy(), m, depth - 1, alpha, beta, move_execution_score), v)

                if v >= beta:
                    return v

                alpha = max(alpha, v)

            return v

    def get_min_scoring_move(self, board: np.ndarray, move: list, depth: int, alpha: int, beta: int, move_execution_score: int) -> int:
        new_state, new_move_score = self.execute_move(move, board.copy())

        if depth == 0:
            # only evaluate state
            return self.score_board(new_state) + new_move_score
        else:
            v = np.inf
            possible_moves = self.calculate_side_possible_moves(False, new_state.copy())

            for m in possible_moves:
                v = min(self.get_max_scoring_move(new_state.copy(), m, depth - 1, alpha, beta, move_execution_score), v)

                if v <= alpha:
                    return v

                beta = min(beta, v)

            return v
