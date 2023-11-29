# Amelia Sinclaire 2023

from enum import StrEnum
import logging
import random

import numpy as np


logging.basicConfig(format='%(message)s', level=logging.WARN)


class State(StrEnum):
    PLAYER_WINS = 'PLAYER'
    COMPUTER_WINS = 'COMPUTER'
    TIE = 'TIE'


class Throws(StrEnum):
    ROCK = 'ROCK'
    PAPER = 'PAPER'
    SCISSORS = 'SCISSORS'


class Rounds:
    def __init__(self, rounds=None):
        self.rounds = rounds if rounds is not None else []

    def add_round(self, r):
        self.rounds.append(r)

    def display_rounds(self):
        for idx, r in enumerate(self.rounds):
            print(f'[{idx}] ({r.p1}, {r.p2}, {r.outcome})')

    def empty(self):
        return len(self.rounds) == 0

    def opponent_last_move(self):
        return self.rounds[-1].p1

    def get_throws_in_outcome(self, computer=True, outcome=State.TIE,
                              previous_n_rounds=None):
        n = 0 if previous_n_rounds is None else -previous_n_rounds
        throws = []
        for r in self.rounds[n:]:
            if r.outcome == outcome:
                if computer:
                    throws.append(r.p2)
                else:
                    throws.append(r.p1)
        return throws

    def get_throws(self, opponent=True, previous_n_rounds=None):
        n = 0 if previous_n_rounds is None else -previous_n_rounds
        throws = []
        for r in self.rounds[n:]:
            if opponent:
                throws.append(r.p1)
            else:
                throws.append(r.p2)
        return throws

    def get_rounds(self, previous_n_rounds=None):
        n = 0 if previous_n_rounds is None else -previous_n_rounds
        rounds = []
        for r in self.rounds[n:]:
            rounds.append(r)
        return rounds

    def percent_outcome(self, outcome, previous_n_rounds=None):
        if self.empty():
            return 0

        return (
            len(self.get_throws_in_outcome(True, outcome, previous_n_rounds))
            / len(self.rounds) * 100)

    def display_percentages(self):
        if self.empty():
            return

        percent_wins = self.percent_outcome(State.PLAYER_WINS)
        percent_ties = self.percent_outcome(State.TIE)
        percent_loss = self.percent_outcome(State.COMPUTER_WINS)

        ratio = (f' | Ratio: {percent_wins/percent_loss:.2f}'
                 if percent_loss > 0 else "")
        print(f'Win: {percent_wins:.2f}% | Lose: {percent_loss:.2f}% | '
              f'Tie: {percent_ties:.2f}%{ratio}')


class Round:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.outcome = evaluate_game(p1, p2)

    def display_round(self):
        print(f'p1: {self.p1}')
        print(f'p2: {self.p2}')
        print(self.outcome.name)

    def display_oneline(self):
        print(f'({self.p1}, {self.p2}, {self.outcome})')


def random_throw():
    return random.choice(list(Throws)[0:3])


def normalize(v):
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def validate_throw(throw):
    throw = throw.upper()
    for t in Throws:
        if throw == t.name or throw == t.name[0]:
            return t
    return None


def evaluate_game(p1, p2):
    if p1 == p2:
        return State.TIE
    if p1 == Throws.ROCK:
        if p2 == Throws.PAPER:
            return State.COMPUTER_WINS
        return State.PLAYER_WINS
    if p1 == Throws.PAPER:
        if p2 == Throws.SCISSORS:
            return State.COMPUTER_WINS
        return State.PLAYER_WINS
    if p2 == Throws.ROCK:
        return State.COMPUTER_WINS
    return State.PLAYER_WINS


"""
# import math
# from RockPaperScissors import Strategies
def main(all_rounds):
    computer = Strategies.SameliaBot(computer=True)  # "computer" / p2
    comp2 = Strategies.BeatMostFreq(computer=False)  # "human" / p1

    def loop(all_rounds, max_rounds=math.inf):
        if max_rounds == 0:
            exit()
        # Get Computer Throw
        # p2_throw = smart_throw(all_rounds)
        p2_throw = computer.throw()

        # Get Player Throw
        p1_throw = None  # comp2.throw()
        while p1_throw is None:
            p1_throw = input("What will you Throw? > ")
            if p1_throw.lower() == 'exit':
                print('exiting program')
                exit()
            p1_throw = validate_throw(p1_throw)
        this_round = Round(p1_throw, p2_throw)
        this_round.display_round()
        all_rounds.add_round(this_round)
        # all_rounds.display_rounds()
        all_rounds.display_percentages()

        computer.update(all_rounds)
        comp2.update(all_rounds)

        loop(all_rounds, max_rounds-1)

    loop(all_rounds)


if __name__ == "__main__":
    all_rounds = Rounds()
    main(all_rounds)
"""
