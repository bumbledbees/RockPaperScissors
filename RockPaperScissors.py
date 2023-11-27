# Amelia Sinclaire 2023
import random
from enum import Enum
import numpy as np
import logging
logging.basicConfig(format='%(message)s', level=logging.ERROR)
import Strategies

State = Enum('State', ['HUMAN_WINS', 'COMPUTER_WINS', 'TIE'])
Throws = Enum('Throw', ['ROCK', 'PAPER', 'SCISSORS', 'EXIT'])


class Rounds:
    rounds = []

    def __init__(self, rounds=None):
        Rounds.rounds = rounds if rounds is not None else []

    def add_round(self, r):
        self.rounds.append(r)

    def display_rounds(self):
        for idx, r in enumerate(self.rounds):
            print(f'[{idx}] ({r.p1.name}, {r.p2.name}, {r.outcome.name})')

    def empty(self):
        return len(self.rounds) == 0

    def opponent_last_move(self):
        return self.rounds[-1].p1

    def get_opponent_throws_in_outcome(self, outcome, previous_n_rounds=None):
        n = 0 if previous_n_rounds is None else -previous_n_rounds
        throws = []
        for r in self.rounds[n:]:
            if r.outcome == outcome:
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

    def get_rounds(self,previous_n_rounds=None):
        n = 0 if previous_n_rounds is None else -previous_n_rounds
        rounds = []
        for r in self.rounds[n:]:
            rounds.append(r)
        return rounds

    def percent_outcome(self, outcome, previous_n_rounds=None):
        if self.empty():
            return 0
        return 100*len(self.get_opponent_throws_in_outcome(outcome, previous_n_rounds)) / len(self.rounds)

    def display_percentages(self):
        if self.empty():
            return
        percent_wins = self.percent_outcome(State['HUMAN_WINS'])
        percent_ties = self.percent_outcome(State['TIE'])
        percent_loss = self.percent_outcome(State['COMPUTER_WINS'])
        print(f'Win: {percent_wins:.2f}% | Lose: {percent_loss:.2f}% | Tie: {percent_ties:.2f}%')


class Round:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.outcome = evaluate_game(p1, p2)

    def display_round(self):
        print(f'p1: {self.p1.name}')
        print(f'p2: {self.p2.name}')
        print(self.outcome.name)

    def display_oneline(self):
        print(f'({self.p1.name}, {self.p2.name}, {self.outcome.name})')


def random_throw():
    return random.choice(list(Throws)[0:3])


def copy_opponent_last_move(rounds):
    if rounds.empty():
        return random_throw()
    return rounds.opponent_last_move()


def beat(throw):
    if throw.value == Throws['ROCK'].value:
        return Throws['PAPER']
    if throw.value == Throws['PAPER'].value:
        return Throws['SCISSORS']
    return Throws['ROCK']


def lose(throw):
    if throw.value == Throws['ROCK'].value:
        return Throws['SCISSORS']
    if throw.value == Throws['PAPER'].value:
        return Throws['ROCK']
    return Throws['PAPER']


def beat_opponent_last_move(rounds):
    if rounds.empty():
        return random_throw()
    return beat(rounds.opponent_last_move())


def normalize(v):
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def detect_pattern(throws, pattern_length):
    # detect patterns of length n (if at least room for 2 repetitions)
    if len(throws) >= pattern_length*2:
        pattern2 = throws[-pattern_length:]
        if pattern2 == throws[-pattern_length*2:-pattern_length]:
            return True, pattern2
    return False, None


def contains_pattern(throws):
    # detect patterns of length n (if at least room for 2 repetitions)
    for n in range(2, 2+(len(throws)//2)):
        is_pattern, pattern = detect_pattern(throws, n)
        if is_pattern:
            return is_pattern, pattern
    return False, None


def smart_throw(rounds):
    if rounds.empty():  # always throw paper first
        return Throws['PAPER']

    choices = [Throws['ROCK'], Throws['PAPER'], Throws['SCISSORS']]
    probabilities = [0, 0, 0]
    opponent_probability = [0, 0, 0]

    # take opponent's wins and calculate freq of each throw that led to that
    wins = rounds.get_opponent_throws_in_outcome(State['HUMAN_WINS'], previous_n_rounds=5)
    win_count = [0, 0, 0]  # R, P, S
    for w in wins:
        if w.value == Throws['ROCK'].value:
            win_count[0] += 1
            continue
        if w.value == Throws['PAPER'].value:
            win_count[1] += 1
            continue
        win_count[2] += 1
    win_count = normalize(win_count)
    opponent_probability = [sum(i) for i in zip(opponent_probability, win_count)]
    # opponent_probability = normalize(opponent_probability)

    # if opponent has won < 33% of previous n games, we predict they will play their
    # least common throws in those n games
    percent_win = rounds.percent_outcome(State['HUMAN_WINS'], previous_n_rounds=5)
    if percent_win < 33:
        losses = rounds.get_opponent_throws_in_outcome(State['COMPUTER_WINS'], previous_n_rounds=5)
        ties = rounds.get_opponent_throws_in_outcome(State['TIE'], previous_n_rounds=5)
        loss_count = [0, 0, 0]  # R, P, S
        for loss in losses+ties:
            if loss.value == Throws['ROCK'].value:
                loss_count[0] -= 1
                continue
            if loss.value == Throws['PAPER'].value:
                loss_count[1] -= 1
                continue
            loss_count[2] -= 1
        loss_count = normalize(loss_count)
        opponent_probability = [sum(i) for i in zip(opponent_probability, loss_count)]

    # # if previous two throws from opponent were identical, they will throw the same again
    # throws = rounds.get_throws(previous_n_rounds=2)
    # if len(list(set(throws))) == 1:
    #     repetitive_throws = [0, 0, 0]
    #     if throws[0] == Throws['ROCK']:
    #         repetitive_throws[0] += 1
    #     elif throws[0] == Throws['PAPER']:
    #         repetitive_throws[1] += 1
    #     else:
    #         repetitive_throws[2] += 1
    #     repetitive_throws = normalize(repetitive_throws)
    #     opponent_probability = [sum(i) for i in zip(opponent_probability, repetitive_throws)]
    #
    # # if previous four throws from opponent were identical, they will not throw the same again
    # throws = rounds.get_throws(previous_n_rounds=4)
    # if len(list(set(throws))) == 1:
    #     repetitive_throws = [0, 0, 0]
    #     if throws[0] == Throws['ROCK']:
    #         repetitive_throws[0] -= 1
    #     elif throws[0] == Throws['PAPER']:
    #         repetitive_throws[1] -= 1
    #     else:
    #         repetitive_throws[2] -= 1
    #     repetitive_throws = normalize(repetitive_throws)
    #     opponent_probability = [sum(i) for i in zip(opponent_probability, repetitive_throws)]

    # get previous n throws, see how many of the last throws were consecutive.
    # the more consecutive throws, the more likely it will be thrown again
    repetition_penalty = 4
    if not rounds.empty():
        throws = rounds.get_throws(previous_n_rounds=10)
        consecutive = -1
        last_throw = throws[-1]
        for t in reversed(throws):
            if t == last_throw:
                consecutive += 1
            else:
                break
        # print(consecutive)
        probability_of_throwing_same = (0.00396853*(consecutive**3)) - (0.0811131*(consecutive**2)) + (0.519033*consecutive) - 0.0220979
        repetitive_throws = [0, 0, 0]
        if last_throw.value == Throws['ROCK'].value:
            repetitive_throws[0] += probability_of_throwing_same
        elif last_throw.value == Throws['PAPER'].value:
            repetitive_throws[1] += probability_of_throwing_same
        else:
            repetitive_throws[2] += probability_of_throwing_same
        # repetitive_throws = normalize(repetitive_throws)
        repetitive_throws = [i * repetition_penalty for i in repetitive_throws]
        opponent_probability = [sum(i) for i in zip(opponent_probability, repetitive_throws)]

    # if MY previous two throws were identical, opponent will expect me to play it again and play to beat it
    throws = rounds.get_throws(opponent=False, previous_n_rounds=2)
    if len(list(set(throws))) == 1:
        repetitive_throws = [0, 0, 0]
        if throws[0].value == Throws['ROCK'].value:
            repetitive_throws[1] += 1
        elif throws[0].value == Throws['PAPER'].value:
            repetitive_throws[2] += 1
        else:
            repetitive_throws[0] += 1
        repetitive_throws = normalize(repetitive_throws)
        opponent_probability = [sum(i) for i in zip(opponent_probability, repetitive_throws)]

    # if previous round was a TIE of x then it is more likely opponent will throw y to beat x
    last_round = rounds.get_rounds(previous_n_rounds=1)[0]
    if last_round.outcome == State['TIE']:
        tie_rule = [0, 0, 0]
        if last_round.p1.value == Throws['ROCK'].value:
            tie_rule[1] += 1
        elif last_round.p1.value == Throws['PAPER'].value:
            tie_rule[2] += 1
        else:
            tie_rule[0] += 1
        tie_rule = normalize(tie_rule)
        opponent_probability = [sum(i) for i in zip(opponent_probability, tie_rule)]

    strategy_penalty = 3

    # check if opponent is employing 'play what I played last' strategy for at least the past 3 rounds
    last_rounds = rounds.get_rounds(previous_n_rounds=4)
    employ_play_last_strat = True if len(last_rounds) > 3 else False
    for idx, r in enumerate(last_rounds):
        i_last_played = last_rounds[idx-1].p2 if idx-1 >= 0 else None
        opponent_played = r.p1
        if i_last_played is None:
            continue
        if i_last_played != opponent_played:
            employ_play_last_strat = False
            break
    if employ_play_last_strat:
        logging.info('STRATEGY DETECTED: YOU ARE PLAYING WHAT I LAST PLAYED')
        play_last = [0, 0, 0]
        i_last_played = last_round.p2
        if i_last_played.value == Throws['ROCK'].value:
            play_last[0] += 1
        elif i_last_played.value == Throws['PAPER'].value:
            play_last[1] += 1
        else:
            play_last[2] += 1
        play_last = normalize(play_last)
        play_last = [i * strategy_penalty for i in play_last]
        opponent_probability = [sum(i) for i in zip(opponent_probability, play_last)]

    # check if opponent is employing 'play what would have beaten my last throw' strategy during the past 3 rounds
    last_rounds = rounds.get_rounds(previous_n_rounds=4)
    employ_beat_last_strat = True if len(last_rounds) > 3 else False
    for idx, r in enumerate(last_rounds):
        i_last_played = last_rounds[idx - 1].p2 if idx - 1 >= 0 else None
        opponent_played = r.p1
        if i_last_played is None:
            continue
        if beat(i_last_played) != opponent_played:
            employ_beat_last_strat = False
            break
    if employ_beat_last_strat:
        logging.info('STRATEGY DETECTED: YOU ARE PLAYING WHAT WOULD\'VE BEATEN WHAT I LAST PLAYED')
        beat_last = [0, 0, 0]
        i_last_played = last_round.p2
        if i_last_played.value == Throws['ROCK'].value:
            beat_last[1] += 1
        elif i_last_played.value == Throws['PAPER'].value:
            beat_last[2] += 1
        else:
            beat_last[0] += 1
        beat_last = normalize(beat_last)
        beat_last = [i * strategy_penalty for i in beat_last]
        opponent_probability = [sum(i) for i in zip(opponent_probability, beat_last)]

    # check if opponent is employing 'play what would have lost to my last throw' strategy during the past 3 rounds
    last_rounds = rounds.get_rounds(previous_n_rounds=4)
    employ_lose_last_strat = True if len(last_rounds) > 3 else False
    for idx, r in enumerate(last_rounds):
        i_last_played = last_rounds[idx - 1].p2 if idx - 1 >= 0 else None
        opponent_played = r.p1
        if i_last_played is None:
            continue
        if (opponent_played) != lose(i_last_played):
            employ_lose_last_strat = False
            break
    if employ_lose_last_strat:
        logging.info('STRATEGY DETECTED: YOU ARE PLAYING WHAT WOULD\'VE LOST TO WHAT I LAST PLAYED')
        lose_last = [0, 0, 0]
        i_last_played = last_round.p2
        if i_last_played.value == Throws['ROCK'].value:
            lose_last[2] += 1
        elif i_last_played.value == Throws['PAPER'].value:
            lose_last[0] += 1
        else:
            lose_last[1] += 1
        lose_last = normalize(lose_last)
        lose_last = [i * strategy_penalty for i in lose_last]
        opponent_probability = [sum(i) for i in zip(opponent_probability, lose_last)]

    # check opponent throws for patterns
    opponent_throws = rounds.get_throws(previous_n_rounds=32)
    pattern_strat, pattern = contains_pattern(opponent_throws)
    if pattern_strat:
        logging.info(f'STRATEGY DETECTED: YOU ARE USING A PATTERN {pattern}')
        pattern_strat_probs = [0, 0, 0]
        next_throw = pattern[0]
        # print(f'pattern detected: {pattern}')
        if next_throw.value == Throws['ROCK'].value:
            pattern_strat_probs[0] += 1
        elif next_throw.value == Throws['PAPER'].value:
            pattern_strat_probs[1] += 1
        else:
            pattern_strat_probs[2] += 1
        pattern_strat_probs = [i * strategy_penalty * len(pattern) for i in pattern_strat_probs]
        opponent_probability = [sum(i) for i in zip(opponent_probability, pattern_strat_probs)]

    # add weights to probabilities, but add the rock weight to the paper probability
    # that is, whatever is most likely to amke the opponent win, we beat that.
    probabilities[1] += opponent_probability[0]  # add ROCK to PAPER
    probabilities[2] += opponent_probability[1]  # add PAPER to SCISSORS
    probabilities[0] += opponent_probability[2]  # add SCISSORS to ROCK

    # print(f'rock: {opponent_probability[0]}, paper: {opponent_probability[1]}, scissors: {opponent_probability[2]}')
    # print(f'rock: {probabilities[0]}, paper: {probabilities[1]}, scissors: {probabilities[2]}')
    index_max = np.argwhere(probabilities == np.amax(probabilities))
    choice = random.choice(index_max)[0]
    return choices[choice]
    # vvv non-deterministic vvv
    # try:
    #     return random.choices(choices, weights=normalize(probabilities), k=1)[0]
    # except:
    #     return random_throw()


def validate_throw(t):
    throw = t.upper()
    for t in Throws:
        if throw == t.name:
            return t
    match throw:
        case "R": return Throws['ROCK']
        case "P": return Throws['PAPER']
        case "S": return Throws['SCISSORS']
        case _: return None


def evaluate_game(p1, p2):
    if p1.value == p2.value:
        return State['TIE']
    if p1.value == Throws['ROCK'].value:
        if p2.value == Throws['PAPER'].value:
            return State['COMPUTER_WINS']
        return State['HUMAN_WINS']
    if p1.value == Throws['PAPER'].value:
        if p2.value == Throws['SCISSORS'].value:
            return State['COMPUTER_WINS']
        return State['HUMAN_WINS']
    if p2.value == Throws['ROCK'].value:
        return State['COMPUTER_WINS']
    return State['HUMAN_WINS']


def main(all_rounds):
    computer = Strategies.BeatLastStrat(computer=True)  # "computer" / p2
    comp2 = Strategies.CopyStrat(computer=False)  # "human" / p1
    import math

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
            p1_throw = validate_throw(p1_throw)
        if p1_throw.value == Throws['EXIT'].value:
            print('exiting program')
            exit()
        this_round = Round(p1_throw, p2_throw)
        this_round.display_round()
        all_rounds.add_round(this_round)
        # all_rounds.display_rounds()
        all_rounds.display_percentages()

        computer.update(all_rounds)
        # comp2.update(all_rounds)

        loop(all_rounds, max_rounds-1)

    loop(all_rounds, max_rounds=100)


if __name__ == "__main__":
    all_rounds = Rounds()
    main(all_rounds)
