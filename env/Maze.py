import numpy as np
from env.AbstractMDP import AbstractMDP

class Maze33(AbstractMDP):
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3

    def __init__(self, success_prob, discount):

        self.n_columns = 3
        self.n_rows = 3

        terminal_states = []
        initial_states = [0]
        self.walls = [{0, 1}, {3, 4}, {1, 2}, {7, 8}]

        goal_state = 2
        P = self.get_transition_matrix(success_prob, terminal_states)
        R = self.get_reward_function(goal_state)

        self._num_states = self.n_columns * self.n_rows
        self._num_actions = 4
        initial_dist = np.zeros((self._num_states))
        for s in initial_states:
            initial_dist[s] = 1 / len(initial_states)

        terminal_states = set(terminal_states)

        super().__init__(P, R, discount, initial_dist)

    def get_transition_matrix(self, success_prob, terminal_states):
        n_states = self.n_columns * self.n_rows

        unif_prob = (1 - success_prob) / 3
        P = np.zeros((4, n_states, n_states))

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

        if action == Maze33.ACTION_UP:
            top_c = column
            top_r = max(row - 1, 0)
            target = top_r * self.n_columns + top_c
        elif action == Maze33.ACTION_RIGHT:
            right_c = min(column + 1, self.n_columns - 1)
            right_r = row
            target = right_r * self.n_columns + right_c
        elif action == Maze33.ACTION_DOWN:
            bottom_c = column
            bottom_r = min(row + 1, self.n_rows - 1)
            target = bottom_r * self.n_columns + bottom_c
        elif action == Maze33.ACTION_LEFT:
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
            for action in range(4):
                if self.get_target(state, action) == goal_state:
                    R[action, state] = 1
        
        return R


