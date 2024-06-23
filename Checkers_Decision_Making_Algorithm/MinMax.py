import numpy as np

def temp_state_score() -> int:
    return np.random.randint(-25, 26)


def calculate_move_min_max(depth: int, max_depth: int, my_game, game_state: np.ndarray, max_player_turn: bool, alpha: int, beta: int) -> tuple:
    # return score, move

    saved_moves = []
    saved_scores = []
    best_move = []

    if depth == 0:
        # now at the deepest tree point
        board_score = my_game.score_board(game_state)
        # return the current game state score
        return board_score, best_move
    else:
        # go down one deeper
        all_piece_moves = my_game.calculate_side_possible_moves(max_player_turn, my_game.board.copy())

        if len(all_piece_moves) > 0:
            # go through moves and call min max

            max_score = -25
            min_score = 25

            end_search = False
            # for each piece that can move
            for p in all_piece_moves:
                if end_search:
                    break

                # for each move of the piece
                for m in p:

                    new_state = my_game.execute_move(m, game_state.copy())

                    if max_player_turn:
                        returned_score, _ = calculate_move_min_max(depth - 1, max_depth, my_game, new_state.copy(),
                                                                   False, alpha, beta)
                    else:
                        returned_score, _ = calculate_move_min_max(depth - 1, max_depth, my_game, new_state.copy(),
                                                                   True, alpha, beta)

                    if depth == max_depth:
                        saved_moves.append(m)
                        saved_scores.append(returned_score)

                    if max_player_turn:
                        max_score = max(max_score, returned_score)
                        alpha = max(alpha, returned_score)
                        if beta <= alpha:
                            end_search = True
                            break

                    else:
                        min_score = min(min_score, returned_score)
                        beta = min(beta, returned_score)
                        if beta <= alpha:
                            end_search = True
                            break

            if depth != max_depth:
                # after checking each child value, returning the child score
                if max_player_turn:
                    return max_score, []
                else:
                    return min_score, []
            else:
                # if at the first minmax call

                if max_player_turn:
                    # looking for max value
                    best_score = max(saved_scores)
                else:
                    # looking for min value
                    best_score = min(saved_scores)

                # save the best moves in the selection list
                best_moves = []
                for score_index in range(len(saved_scores)):
                    if saved_scores[score_index] == best_score:
                        best_moves.append(saved_moves[score_index])

                # select one of best moves at random
                selected_move = np.random.randint(0, len(best_moves))

                # OR select the first equally best move
                # selected_move = 0

                print(saved_scores, saved_moves)

                # only returning the best move
                return best_score, best_moves[selected_move]

        else:
            # no possible moves for current player
            return my_game.score_board(game_state), best_move
