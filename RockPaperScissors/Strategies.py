# Amelia Sinclaire 2023

import logging
import random

import numpy as np

from RockPaperScissors.Game import normalize, Rounds, State, Throws


class Strategy:
    def __init__(self, rounds=None, computer=True):
        self.computer = computer
        self.rounds = rounds if rounds is not None else Rounds()

    def update(self, rounds):
        self.rounds = rounds

    def throw(self):
        pass

    @staticmethod
    def counter_throw(throw):
        if throw.value == Throws['ROCK'].value:
            return Throws['PAPER']
        if throw.value == Throws['PAPER'].value:
            return Throws['SCISSORS']
        return Throws['ROCK']

    def counter_strat(self, strat, lookback=10):
        if self.rounds.empty():  # no info = no confidence
            return {
                "opponent_next_throw": Throws['PAPER'],
                "frequency": 0,
                "consecutive": 0,
            }
        # for previous `lookback` number of rounds
        # see how many of them the given strat was employed by opp
        strat_count = 0
        consecutive = 0
        consecutive_done = False
        for i in range(0, lookback):
            n = 0 if lookback is None else -lookback
            if i == 0:
                lookback_rounds = self.rounds.rounds[n - i:]
            else:
                lookback_rounds = self.rounds.rounds[n - i:-i]
            if len(lookback_rounds) == 0:
                continue
            current_round = lookback_rounds[-1]
            del lookback_rounds[-1]

            generator = strat(computer=(not self.computer),
                              rounds=Rounds(lookback_rounds))
            expected = generator.throw()

            if self.computer:
                actual = current_round.p1
            else:
                actual = current_round.p2

            if expected.value == actual.value:
                # print('*bing!*')
                strat_count += 1
                if not consecutive_done:
                    consecutive += 1
                continue
            consecutive_done = True

        frequency = strat_count / lookback
        g = strat(computer=(not self.computer), rounds=self.rounds)
        opponent_next_throw = g.throw()
        logging.info(
            f'{g.__class__.__name__} FREQUENCY: {frequency} |CONSECUTIVE: '
            f'{consecutive} | OPPONENT NEXT THROW : {opponent_next_throw}')

        output = {
            "opponent_next_throw": opponent_next_throw,
            "frequency": frequency,
            "consecutive": consecutive,
        }
        return output

    @staticmethod
    def detect_pattern(opponent_throws, pattern_length):
        count = 0
        pattern_found = False
        max_reps = len(opponent_throws) // pattern_length
        pattern = opponent_throws[-pattern_length:]
        # print(f'max_reps = {max_reps}')
        for i in range(1, max_reps):
            # print(f'rep={i}')
            end_index = -(i*pattern_length)
            end_index = len(opponent_throws) if end_index == 0 else end_index
            start_index = end_index - pattern_length
            # print(f'[{start_index}:{end_index}]')
            potential_pattern = opponent_throws[start_index:end_index]
            if len(potential_pattern) != pattern_length:
                # print('invalid potential pattern (out of bounds)')
                continue
            # print(f'checking if {pattern} == potential: {potential_pattern}')
            if pattern == potential_pattern:
                # print('*bing*')
                count += 1
                pattern_found = True
                continue
            if pattern_found:  # we found some pattern, but it broke.
                return True, pattern, count  # (early exit)
            # if here then we didn't find a pattern
            return False, pattern, 0
        if pattern_found:  # we found some pattern, and it never it broke.
            return True, pattern, count
        # no pattern ever found
        return False, None, 0

    def contains_patterns(self, lookback=None):
        n = 0 if lookback is None else -lookback
        if self.computer:
            opponent_throws = [r.p1 for r in self.rounds.rounds[n:]]
        else:
            opponent_throws = [r.p2 for r in self.rounds.rounds[n:]]
        # detect patterns of length n (if at least room for 2 repetitions)
        all_found_patterns = []
        for n in range(1, 1 + (len(opponent_throws) // 2)):
            # print(f'checking patterns of length {n}')
            is_pattern, pattern, consecutive = (
                self.detect_pattern(opponent_throws, n))
            if is_pattern:
                all_found_patterns.append((pattern, consecutive))
        return all_found_patterns

    @staticmethod
    def most_frequent(throws):
        counter = 0
        most_freq = throws[0]
        for r in throws:
            curr_frequency = throws.count(r)
            if curr_frequency > counter:
                counter = curr_frequency
                most_freq = r
        return most_freq


class RandomStrat(Strategy):
    def throw(self):
        return random.choice(list(Throws)[0:3])  # random throw


class CopyLast(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        if self.computer:
            return self.rounds.rounds[-1].p1  # i am p2, copy p1's last move
        return self.rounds.rounds[-1].p2  # i am p1, copy p2's last move


class BeatLast(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        if self.computer:
            # i am p2, beat p1's last move
            return self.counter_throw(self.rounds.rounds[-1].p1)
        # i am p1, beat p2's last move
        return self.counter_throw(self.rounds.rounds[-1].p2)


class LoseLast(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        last_round = self.rounds.rounds[-1]
        if self.computer:
            # i am p2, lose to p1's last move
            return self.counter_throw(self.counter_throw(last_round.p1))
        # i am p1, lose to p2's last move
        return self.counter_throw(self.counter_throw(last_round.p2))


class SwitchAfterTwo(Strategy):
    def throw(self):
        if len(self.rounds.rounds) < 2:
            return random.choice(list(Throws)[0:3])  # random throw
        # assume if I have thrown the same thing for the last two turns,
        # I won't throw the same again
        p1_last_move = self.rounds.rounds[-1].p1
        p1_second_last_move = self.rounds.rounds[-2].p1
        p2_last_move = self.rounds.rounds[-1].p2
        p2_second_last_move = self.rounds.rounds[-2].p2
        if self.computer:
            same_last_move = p2_last_move.value == p2_second_last_move.value
            last_move = p2_last_move
        else:
            same_last_move = p1_last_move.value == p1_second_last_move.value
            last_move = p1_last_move
        if same_last_move:
            # ex. if I throw scissors twice in a row, I won't throw it again.
            # rather I'll throw ROCK or PAPER, preferring the next in the cycle
            return self.counter_throw(last_move)
        # if not last two same, I throw randomly
        return random.choice(list(Throws)[0:3])  # random throw


class StickToWins(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        # if I just won the previous round, I should throw the same thing again
        p1_last_move = self.rounds.rounds[-1].p1
        p2_last_move = self.rounds.rounds[-1].p2
        if self.computer:
            won_last = (self.rounds.rounds[-1].outcome.value
                        == State['COMPUTER_WINS'].value)
            last_move = p2_last_move
        else:
            won_last = (self.rounds.rounds[-1].outcome.value
                        == State['PLAYER_WINS'].value)
            last_move = p1_last_move
        if won_last:
            return last_move
        # if I didn't win last round, throw whatever got me the most wins
        # previously
        if self.computer:
            wins = self.rounds.get_throws_in_outcome(self.computer,
                                                     State['COMPUTER_WINS'])
        else:
            wins = self.rounds.get_throws_in_outcome(self.computer,
                                                     State['PLAYER_WINS'])
        if len(wins) == 0:  # if I haven't won, then throw randomly
            return random.choice(list(Throws)[0:3])  # random throw

        # Program to find most frequent element in a list
        return self.most_frequent(wins)


class ChangeIfLoss(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        # if I just LOST the previous round, I should NOT throw
        # the same thing again
        p1_last_move = self.rounds.rounds[-1].p1
        p2_last_move = self.rounds.rounds[-1].p2
        if self.computer:
            lost_last = (self.rounds.rounds[-1].outcome.value
                         == State['COMPUTER_WINS'].value)
            last_move = p2_last_move
        else:
            lost_last = (self.rounds.rounds[-1].outcome.value
                         == State['PLAYER_WINS'].value)
            last_move = p1_last_move
        lost_or_tie_last = (lost_last or self.rounds.rounds[-1].outcome.value
                            == State['TIE'].value)
        if lost_or_tie_last:
            return self.counter_throw(last_move)
        # if I didn't lose or tie last round, throw random
        return random.choice(list(Throws)[0:3])  # random throw


class ChangeIfTie(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        # if I just TIEd the previous round, I will play what would've beaten
        # the TIE
        p1_last_move = self.rounds.rounds[-1].p1
        p2_last_move = self.rounds.rounds[-1].p2
        if self.computer:
            last_move = p2_last_move
        else:
            last_move = p1_last_move
        tie_last = self.rounds.rounds[-1].outcome.value == State['TIE'].value
        if tie_last:
            return self.counter_throw(last_move)
        # if I didn't lose or tie last round, throw random
        return random.choice(list(Throws)[0:3])  # random throw


class StatisticalStrat(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        choices = (list(Throws)[0:3])
        weights = [0.354, 0.296, 0.350]
        # random frequencies I found online:
        # https://www.quora.com/Is-there-any-research-or-data-into-what-people-usually-throw-first-in-a-game-of-Rock-Paper-Scissors
        weights = normalize(weights)
        # non-deterministically sample from the choices by their frequencies
        opponent_throw = random.choices(choices, weights=weights, k=1)[0]
        # defeat the likely throw
        return self.counter_throw(opponent_throw)


class BeatMostFreq(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        # look at all opponent's throws
        # counteract whatever they throw most often
        opponent_throws = self.rounds.get_throws(opponent=(not self.computer))
        return self.counter_throw(self.most_frequent(opponent_throws))


class SameliaBot(Strategy):
    def throw(self):
        copy = self.counter_strat(CopyLast, lookback=10)
        beat_last = self.counter_strat(BeatLast, lookback=10)
        lose_last = self.counter_strat(LoseLast, lookback=10)
        switch_after_two = self.counter_strat(SwitchAfterTwo, lookback=10)
        stick_to_wins = self.counter_strat(StickToWins)
        loss_change = self.counter_strat(ChangeIfLoss)
        tie_change = self.counter_strat(ChangeIfTie)
        statistical = self.counter_strat(StatisticalStrat)
        beat_most_freq = self.counter_strat(BeatMostFreq)
        all_patterns = self.contains_patterns()

        opponent_probability = [0.05, 0, 0.01]  # add tiny bias factors

        copy_weight = copy['consecutive']
        if copy['consecutive'] > 2:
            logging.warning('STRATEGY DETECTED: CopyLast')
            opponent_probability[copy['opponent_next_throw'].value-1] += copy_weight

        lose_last_weight = lose_last['consecutive']
        if lose_last['consecutive'] > 2:
            logging.warning('STRATEGY DETECTED: LoseLast')
            opponent_probability[lose_last['opponent_next_throw'].value-1] += lose_last_weight

        beat_last_weight = beat_last['consecutive']
        if beat_last['consecutive'] > 2:
            logging.warning('STRATEGY DETECTED: BeatLast')
            opponent_probability[beat_last['opponent_next_throw'].value - 1] += beat_last_weight

        switch_after_two_weight = 1 + switch_after_two['consecutive'] * 0.8
        if switch_after_two['frequency'] > 0.2:
            logging.warning('STRATEGY DETECTED: SwitchAfterTwo')
            opponent_probability[switch_after_two['opponent_next_throw'].value-1] += switch_after_two_weight

        stick_to_wins_weight = 1 + stick_to_wins['frequency']
        if stick_to_wins['frequency'] > 0.6:
            logging.warning('STRATEGY DETECTED: StickToWins')
            opponent_probability[stick_to_wins['opponent_next_throw'].value-1] += stick_to_wins_weight

        loss_change_weight = 1 + loss_change['frequency']
        if loss_change['frequency'] > 0.6:
            logging.warning('STRATEGY DETECTED: ChangeIfLoss')
            opponent_probability[loss_change['opponent_next_throw'].value-1] += loss_change_weight

        tie_change_weight = 1 + tie_change['frequency']
        if tie_change['frequency'] > 0.6:
            logging.warning('STRATEGY DETECTED: ChangeIfTie')
            opponent_probability[tie_change['opponent_next_throw'].value - 1] += tie_change_weight

        statistical_weight = 1 + statistical['frequency'] * 0.5
        if statistical['frequency'] > 0.75:
            logging.warning('STRATEGY DETECTED: StatisticalStrat')
            opponent_probability[statistical['opponent_next_throw'].value-1] += statistical_weight

        beat_most_freq_weight = 1 + beat_most_freq['frequency'] * 0.5
        if beat_most_freq['frequency'] > 0.6:
            logging.warning('STRATEGY DETECTED: BeatMostFreq')
            opponent_probability[beat_most_freq['opponent_next_throw'].value - 1] += beat_most_freq_weight

        if len(all_patterns) > 0:
            all_patterns = reversed(sorted(all_patterns,
                                           key=lambda x: (len(x[0]), x[1])))

            for p in all_patterns:
                pattern, consecutive = p
                pattern_weight = (
                    1.5 * (((1 + consecutive) + (len(pattern) ** 2) * 0.05)
                           / len(all_patterns)))
                logging.warning(f'STRATEGY DETECTED: Pattern {pattern} '
                                f': {consecutive}')
                opponent_next_throw = pattern[0]
                opponent_probability[opponent_next_throw.value-1] += pattern_weight

        opponent_probability = normalize(opponent_probability)
        # logging.warning(opponent_probability)
        index_max = np.argwhere(opponent_probability
                                == np.amax(opponent_probability))
        opponent_next_throw = list(Throws)[random.choice(index_max)[0]]
        # counter opponent's most likely move
        return self.counter_throw(opponent_next_throw)


def pretty_print(rounds):
    for r in rounds:
        r.display_oneline()
