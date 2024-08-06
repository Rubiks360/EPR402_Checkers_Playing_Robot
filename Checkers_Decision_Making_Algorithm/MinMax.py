import numpy as np

ALPHA_BETA_ENABLE = True

def temp_state_score() -> int:
    return np.random.randint(-25, 26)


def calculate_move_min_max(depth: int, my_game, game_state: np.ndarray, max_player_turn: bool, alpha: int, beta: int, search_score: int) -> tuple:
    # return score, move
    # depth = 0, just look at current state
    # depth = 1, look af first possible moves and game states thereafter
    # depth = n, look at n-1 deep possible moves ...

    saved_moves = []
    saved_scores = []
    best_move = []

    if depth == 0:
        # now at the deepest tree point
        board_score = my_game.score_board(game_state)
        # return the current game state score
        return (board_score + search_score), best_move
    else:
        # go down one possible move deeper
        all_piece_moves = my_game.calculate_side_possible_moves(max_player_turn, my_game.board.copy())

        if len(all_piece_moves) > 0:
            # go through moves and call min max

            best_score = 0

            end_search = False
            # for each piece that can move
            for p in all_piece_moves:
                if end_search:
                    break

                # for each move of the piece
                for m in p:
                    new_state, new_move_score = my_game.execute_move(m, game_state.copy())

                    total_score = new_move_score + search_score

                    if max_player_turn:
                        best_score = -np.inf
                        returned_score, _ = calculate_move_min_max(depth - 1, my_game, new_state.copy(),
                                                                   False, alpha, beta, total_score)

                        best_score = max(best_score, returned_score)
                    else:
                        best_score = np.inf
                        returned_score, _ = calculate_move_min_max(depth - 1, my_game, new_state.copy(),
                                                                   True, alpha, beta, total_score)

                        best_score = min(best_score, returned_score)

                    if depth == 1:
                        # save scores of the first possible moves of the search
                        saved_moves.append(m)
                        saved_scores.append(best_score)

                    if max_player_turn:
                        alpha = max(alpha, best_score)
                        if (beta <= alpha) & (ALPHA_BETA_ENABLE is True):
                            end_search = True
                            break

                    else:
                        beta = min(beta, best_score)
                        if (beta <= alpha) & (ALPHA_BETA_ENABLE is True):
                            end_search = True
                            break

            if depth > 1:
                '''
                after looking at all possible moves, 
                if still looking at the first possible moves save the moves + scores
                else only return the max or min score
                '''
                return best_score, []
            else:
                # currently at max search depth

                if max_player_turn:
                    # looking for max value
                    best_possible_score = max(saved_scores)
                else:
                    # looking for min value
                    best_possible_score = min(saved_scores)

                # save the best moves in the selection list
                best_moves = []
                for score_index in range(len(saved_scores)):
                    if saved_scores[score_index] == best_possible_score:
                        best_moves.append(saved_moves[score_index])

                # select one of best moves at random
                # selected_move = np.random.randint(0, len(best_moves))

                # OR select the first equally best move
                selected_move = 0

                # print(saved_scores, saved_moves)

                # only returning the best move
                return best_possible_score, best_moves[selected_move]
        else:
            # no possible moves for current player
            return (my_game.score_board(game_state) + search_score), best_move
