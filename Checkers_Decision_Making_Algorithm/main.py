from CheckersBoard import *

if __name__ == '__main__':
    my_board = CheckersBoard()

    computer_turn = False
    game_end = False

    print("Player is colour Black\n")

    while not game_end:
        my_board.print_board_layout()
        if computer_turn:
            print("Computer turn")

            all_piece_moves = my_board.calculate_side_possible_moves(True, my_board.board.copy())
            max = -1000
            max_moves = []
            if len(all_piece_moves) > 0:
                # my_board.print_moves(all_piece_moves)
                all_piece_moves_score = []
                for p in all_piece_moves:
                    for m in p:
                        all_piece_moves_score.append(my_board.score_board_after_moves(m, my_board.board.copy()))

                        if max == all_piece_moves_score[-1]:
                            max_moves.append(m)
                        elif max < all_piece_moves_score[-1]:
                            max = all_piece_moves_score[-1]
                            max_moves = [m]

                # select one of the scores that maxes game
                move_num = np.random.randint(0, len(max_moves))
                my_board.execute_move(max_moves[move_num])
            else:
                print("No moves available for computer, player wins!!!")
                game_end = True

            computer_turn = False
        else:
            input_player = input("Player turn, ender moves (r, c, ...): ")

            r_c_moves = input_player.split()

            moves = []
            for i in range(0, len(r_c_moves), 2):
                moves.append([int(r_c_moves[i]), int(r_c_moves[i + 1])])

            if len(moves) > 0:
                my_board.execute_move(moves)
            else:
                print("Player surrendered, computer wins!!!")
                game_end = True

            computer_turn = True

        print("===========================================================")
        print()
        print()
        print()
        print("===========================================================")

    print("GAME OVER")

