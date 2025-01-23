import os

import numpy as np
import yaml
from scipy.stats import rv_discrete

from env.AbstractMDP import AbstractMDP



class CliffWalk(AbstractMDP):
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    def __init__(self, success_prob, discount):

        self.n_columns = 6
        self.n_rows = 6

        terminal_states = [self.n_columns-1]

        for state in range(self.n_columns * self.n_rows):
            if state // self.n_columns in [0,2,4] and state % self.n_columns in range(1, self.n_columns-1):
                terminal_states.append(state)

        initial_states = [0]
        self.walls = []
        self.terminal_states = terminal_states

        goal_state = self.n_columns-1
        P = self.get_transition_matrix(success_prob, terminal_states)
        R = self.get_reward_function(goal_state)

        self._num_states = self.n_columns * self.n_rows
        self._num_actions = 4
        initial_dist = np.zeros((self._num_states))
        for s in initial_states:
            initial_dist[s] = 1 / len(initial_states)

        super().__init__(P, R, discount, initial_dist)

    def get_transition_matrix(self, success_prob, terminal_states):
        n_states = self.n_columns * self.n_rows
        P = np.zeros((4, n_states, n_states))
        unif_prob = (1 - success_prob) / 3
        for r in range(self.n_rows):
            for c in range(self.n_columns):
                state = r * self.n_columns + c
                if state in terminal_states:
                    P[:, state, state] = 1
                else:
                    for a in range(4):
                        for dir in range(4):
                            target = self.get_target(state, dir)
                            if dir == a:
                                P[a, state, target] += success_prob
                            else:
                                P[a, state, target] += unif_prob


        return P

    def get_target(self, state, action):
        column = state % self.n_columns
        row = int((state - column) / self.n_columns)

        if action == CliffWalk.ACTION_UP:
            top_c = column
            top_r = max(row - 1, 0)
            target = top_r * self.n_columns + top_c
        elif action == CliffWalk.ACTION_RIGHT:
            right_c = min(column + 1, self.n_columns - 1)
            right_r = row
            target = right_r * self.n_columns + right_c
        elif action == CliffWalk.ACTION_DOWN:
            bottom_c = column
            bottom_r = min(row + 1, self.n_rows - 1)
            target = bottom_r * self.n_columns + bottom_c
        elif action == CliffWalk.ACTION_LEFT:
            left_c = max(column - 1, 0)
            left_r = row
            target = left_r * self.n_columns + left_c
        else:
            raise Exception("Illegal action")

        if {state, target} in self.walls:
            target = state

        return target

    def get_reward_function(self, goal_state):
        n_states = self.n_columns * self.n_rows

        R = np.zeros((4, n_states))

        for state in range(n_states):
            if state in self.terminal_states:
                if state == goal_state:
                    R[:, state] = 20
                elif state % self.n_columns in range(1, self.n_columns-1):
                    if state // self.n_columns == 0:
                        R[:, state] = -32
                    if state // self.n_columns == 2:
                        R[:, state] = -16
                    if state // self.n_columns == 4:
                        R[:, state] = -8
                    if state // self.n_columns == 6:
                        R[:, state] = -4
                    if state // self.n_columns == 8:
                        R[:, state] = -2
                    if state // self.n_columns == 10:
                        R[:, state] = -1
            else:
                R[:, state] = -1
        return R