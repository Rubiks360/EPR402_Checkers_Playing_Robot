import numpy as np

from CheckersBoard import *
from MinMax import *
import time


if __name__ == '__main__':
    my_game = CheckersBoard()

    game_end = False

    computer_light_side = False

    if computer_light_side:
        print("Player is colour Black\nPlayer moves first")
        computer_turn = False
    else:
        print("Player is colour White\nComputer moves first")
        computer_turn = True

    my_game.get_best_move_alpha_beta_search(2, computer_light_side)
    print("TEST#########################")

    while not game_end:
        my_game.print_board_layout(my_game.board)
        if computer_turn:
            print("Computers turn")
            start = time.time()
            # _, moves = calculate_move_min_max(5, my_game, my_game.board.copy(), computer_light_side, -np.inf, np.inf, 0)
            _, moves = my_game.get_best_move_alpha_beta_search(2, computer_light_side)
            # print("Computer move score =", _)
            end = time.time()
            print("Computer found its best move in", end - start, "s")

            if len(moves) > 0:
                print("Computer move:", moves)
                my_game.board, _ = my_game.execute_move(moves, my_game.board)
            else:
                print("Computer has no more moves, player wins!!!")
                game_end = True

            computer_turn = False
        else:
            possible_moves = my_game.calculate_side_possible_moves(not computer_light_side, my_game.board.copy())
            if my_game.check_game_end(my_game.board) | (len(possible_moves) == 0):
                print("Player has no possible moves left.\nPlayer Lost, computer wins!!!")
                game_end = True
            else:
                legal_player_move = False

                while not legal_player_move:
                    input_player = input("Player turn, ender moves (r1, c1, r2, c2, ...):\n")

                    r_c_moves = input_player.split()
                    moves = []
                    if len(r_c_moves) % 2 == 0:
                        for i in range(0, len(r_c_moves), 2):
                            moves.append([int(r_c_moves[i]), int(r_c_moves[i + 1])])
                    else:
                        # give a default error move
                        moves.append([0, 0])

                    if len(moves) > 0:
                        if moves in possible_moves:
                            legal_player_move = True

                        if legal_player_move:
                            print("Player move is legal!")
                            my_game.board, _ = my_game.execute_move(moves, my_game.board)
                            computer_turn = True
                        else:
                            print("Player move illegal, try another move!")
                    else:
                        print("Player surrendered, computer wins!!!")
                        legal_player_move = True
                        game_end = True

        print("===========================================================\n\n\n")
        print("===========================================================")

    print("GAME OVER")

