import numpy as np

def temp_state_score() -> int:
    return np.random.randint(-25, 26)


def calculate_move_min_max(depth: int, max_depth: int, my_game, game_state: np.ndarray, max_player_turn: bool) -> int:
    # return score, move

    saved_scores = []

    if depth == 0:
        # now at the deepest tree point
        # return the current game state score
        return temp_state_score()
    else:
        # go down one deeper
        # calculate all possible moves from the current state
        all_piece_moves = my_game.calculate_side_possible_moves(max_player_turn, my_game.board.copy())

        if len(all_piece_moves) > 0:
            # go through moves and call min max

            max_score = -25
            min_score = 25

            # for each piece that can move
            for p in all_piece_moves:

                # for each move of the piece
                for m in p:

                    new_state = my_game.execute_move(m, game_state.copy())

                    if max_player_turn:
                        returned_score = calculate_move_min_max(depth - 1, max_depth, my_game, new_state.copy(), False)
                    else:
                        returned_score = calculate_move_min_max(depth - 1, max_depth, my_game, new_state.copy(), True)

                    if depth == max_depth:
                        saved_scores.append(returned_score)

                    if max_player_turn:
                        max_score = max(max_score, returned_score)
                    else:
                        min_score = min(min_score, returned_score)

            if depth != max_depth:
                # after checking each child value, returning the child score
                if max_player_turn:
                    return max_score
                else:
                    return min_score
            else:
                # if at the first minmax call

                # find move equal to best return move
                if max_player_turn:
                    # looking for max value
                    best_score = max(saved_scores)
                else:
                    # looking for min value
                    best_score = min(saved_scores)

                print("All scores =", saved_scores)
                print("best score =", best_score)

                # only returning the best move
                return best_score

        else:
            # no possible moves for current player
            return temp_state_score()
