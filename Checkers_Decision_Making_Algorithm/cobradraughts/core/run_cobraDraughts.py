'''
Created on Jul 22, 2011

@author: Davide Aversa
'''

import sys
import os.path

from cobradraughts.core.DraughtsBrain import DraughtsBrain

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

# This is an usage example. It's easy!
D = DraughtsBrain(weights1, 2, weights2, verbose=True)
# D.run_self()

# for i in range(1):
#     my_best_move = D.best_move()
#
#     print(my_best_move)
#     print(type(my_best_move))
#
#     print(my_best_move.type)
#     print(my_best_move.source)
#     print(my_best_move.destination)
#     print(my_best_move.captured)
#     print(my_best_move.promote)
#     print(my_best_move.next)
#
#     '''
#     MOVE :: <5 , 1> -> <4 , 0> { None }
#     <class 'cobradraughts.core.DAction.DAction'>
#     MOVE
#     (5, 1)
#     (4, 0)
#     None
#     False
#     None
#     '''
#
#     D.apply_action(my_best_move)

for i in range(10):
    D.run_self()
    print("The winner is %s!" % D.winner)

    D.reset()


