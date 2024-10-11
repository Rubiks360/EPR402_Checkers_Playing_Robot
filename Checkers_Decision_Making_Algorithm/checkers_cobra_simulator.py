# project chekers AI
from CheckersBoard import *
from MinMax import *
# open source chekcers AI
from cobradraughts.core.DraughtsBrain import DraughtsBrain
from cobradraughts.core.DAction import DAction

import time

weights1 = {'PIECE':400,
          'KING':1200,
          'BACK':10,
          'KBACK':10,
          'CENTER':30,
          'KCENTER':30,
          'FRONT':60,
          'KFRONT':60,
          'MOB':0}

weights2 = {'PIECE':400,
          'KING':800,
          'BACK':40,
          'KBACK':40,
          'CENTER':40,
          'KCENTER':40,
          'FRONT':60,
          'KFRONT':60,
          'MOB':0}

# depth = 0 to 4
my_ai_search_depth = 3
# depth = 1 to 5
opponent_ai_search_depth = 1

# initialise AIs
my_game = CheckersBoard()
my_ai_name = "Home-AI"

opponent_AI = DraughtsBrain(weights1, opponent_ai_search_depth, weights2, verbose=True)
opponent_ai_name = "Cobra-AI"

print("Start Simulation")

# simulation statistics
NUM_GAMES = 1000
num_my_ai_wins = 0
num_opponent_ai_wins = 0
num_draws = 0
num_error_games = 0

print_output = False

time_all_game_start = time.time()

total_time_per_move = 0
total_number_of_moves = 0

# run multiple games
for n_game in range(NUM_GAMES):
    my_game.reset_board()
    opponent_AI.reset()

    game_end = False
    game_end_message = ''

    # alternate what AI begins
    if n_game % 2 == 0:
        computer_light_side = True
    else:
        computer_light_side = False
    ''' OR '''
    # computer_light_side = False

    if computer_light_side:
        print(opponent_ai_name, " is colour Black\n",
              my_ai_name, " is colour White\n",
              opponent_ai_name, " moves first") if print_output else None
        computer_turn = False
        opponent_AI.turn = 'LIGHT'
    else:
        print(my_ai_name, " is colour Black\n",
              opponent_ai_name, " is colour White\n",
              my_ai_name, " moves first") if print_output else None
        computer_turn = True
        opponent_AI.turn = 'BLACK'

    # print staring boards
    print("Starting boards:") if print_output else None
    my_game.print_board_layout(my_game.board) if print_output else None
    print(opponent_AI.board) if print_output else None

    while not game_end:
        if computer_turn:
            print(my_ai_name, "turn") if print_output else None

            time_move_start = time.time()
            _, moves = my_game.get_best_move_alpha_beta_search(my_ai_search_depth, computer_light_side)
            time_move_end = time.time()

            total_time_per_move += time_move_end - time_move_start
            total_number_of_moves += 1

            if len(moves) > 0:
                print(my_ai_name, "move:", moves) if print_output else None
                my_game.board, _ = my_game.execute_move(moves, my_game.board)

                # create cobra-AI action to update the board by executing computer move
                opponent_AI_move = DAction("Move", (moves[0][0], moves[0][1]), (moves[1][0], moves[1][1]), None, False)

                cobra_move_destination = (moves[1][0], moves[1][1])

                if abs(moves[0][0] - moves[1][0]) == 2:
                    # capture by computer
                    opponent_AI_move.type = 'CAPTURE'
                    opponent_AI_move.captured = opponent_AI.board.get_piece(
                        (moves[1][0] + moves[0][0]) >> 1,
                        (moves[1][1] + moves[0][1]) >> 1
                    )  # captured piece

                    for i in range(1, len(moves)-1):
                        new_action = DAction('CAPTURE', (moves[i][0], moves[i][1]), (moves[i+1][0], moves[i+1][1]), None, False)
                        new_action.captured = opponent_AI.board.get_piece(
                            (moves[i+1][0] + moves[i][0]) >> 1,
                            (moves[i+1][1] + moves[i][1]) >> 1
                        )  # captured piece coordinates
                        opponent_AI_move._append_capture(new_action)

                        cobra_move_destination = (moves[i+1][0], moves[i+1][1])
                    # CAPTURE :: <6 , 5> -> <4 , 3> { CAPTURE :: <4 , 3> -> <2 , 1> { None } }
                else:
                    opponent_AI_move.type = 'MOVE'
                    opponent_AI_move.captured = False

                opponent_AI.apply_action(opponent_AI_move)

                opponent_AI.switch_turn()
            else:
                game_end_message = opponent_ai_name
                game_end = True

            computer_turn = False
        else:
            print(opponent_ai_name, "turn") if print_output else None

            if my_game.check_game_end(my_game.board):
                game_end_message = my_ai_name
                game_end = True
            else:
                cobra_best_move = opponent_AI.best_move()

                if cobra_best_move is not None:
                    # translate coordinates
                    move_translate = [[cobra_best_move.source[0], cobra_best_move.source[1]], [cobra_best_move.destination[0], cobra_best_move.destination[1]]]
                    temp_move = cobra_best_move
                    while temp_move.next:
                        temp_move = temp_move.next
                        move_translate.append([temp_move.destination[0], temp_move.destination[1]])
                    print(opponent_ai_name, "trans move:", move_translate) if print_output else None

                    possible_moves = my_game.calculate_side_possible_moves(not computer_light_side, my_game.board.copy())
                    legal_player_move = False

                    if move_translate in possible_moves:
                        legal_player_move = True
                else:
                    legal_player_move = False

                if legal_player_move:
                    opponent_AI.apply_action(cobra_best_move)
                    opponent_AI.switch_turn()
                    # execute Cobra-AI move on game engine
                    my_game.board, _ = my_game.execute_move(move_translate, my_game.board)
                else:
                    if not cobra_best_move:
                        # opponent has no more moves, my-ai wins
                        game_end_message = my_ai_name
                    else:
                        print("PROBLEM, ILLEGAL MOVE!!!!")
                        my_game.print_board_layout(my_game.board)
                        print(opponent_AI.board)
                        print(cobra_best_move)
                        print(move_translate)
                        print()

                        game_end_message = "Game error"
                    game_end = True

            computer_turn = True

        if not game_end:
            print("Boards after move:") if print_output else None
            my_game.print_board_layout(my_game.board) if print_output else None
            print(opponent_AI.board) if print_output else None
            print("===========================================================\n\n\n") if print_output else None
            print("===========================================================") if print_output else None

            if opponent_AI.gameover:
                if computer_light_side:
                    if opponent_AI.winner == 'LIGHT':
                        game_end_message = opponent_ai_name
                    else:
                        game_end_message = my_ai_name
                else:
                    if opponent_AI.winner == 'DARK':
                        game_end_message = opponent_ai_name
                    else:
                        game_end_message = my_ai_name
                game_end = True

            if opponent_AI.nocapturecounter > 50:
                game_end_message = "Draw"
                game_end = True

    if game_end_message == 'Draw':
        num_draws += 1
        print(n_game, "Game is a Draw")
    elif game_end_message == my_ai_name:
        num_my_ai_wins += 1
        print(n_game, "Winner is", my_ai_name)
    elif game_end_message == opponent_ai_name:
        num_opponent_ai_wins += 1
        print(n_game, "Winner is", opponent_ai_name)
    else:
        num_error_games += 1

time_all_game_end = time.time()

print()
print("===================================")
print("Home AI depth =", my_ai_search_depth)
print("Cobra AI depth =", opponent_ai_search_depth)
print()
print("Number of games =", NUM_GAMES)
print("Number of draws =", num_draws)
print("Number of", my_ai_name, "wins =", num_my_ai_wins)
print("Number of", opponent_ai_name, "wins =", num_opponent_ai_wins)
print("Number ERRORS GAMES =", num_error_games)

if (NUM_GAMES - num_error_games) == 0:
    print("Only error games")
else:
    print("Win/Draw rate = ", ((num_draws + num_my_ai_wins) / (NUM_GAMES - num_error_games) * 100), "%")

print()
print("Simulation time =", time_all_game_end - time_all_game_start, "s")
# print("Time per game =", (time_all_game_end - time_all_game_start) / NUM_GAMES, "s")

print("Average number of moves per game =", total_number_of_moves / NUM_GAMES)
print("Average time per move =", total_time_per_move / total_number_of_moves, "s")

