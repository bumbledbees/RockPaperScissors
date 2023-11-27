# Amelia Sinclaire 2023
import random
from RockPaperScissors import Throws, Rounds
import logging


class Strategy:
    def __init__(self, computer=True, rounds=None):
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

    def counter_strat(self, strat, lookback=10, weight=1, consecutive_freq_thresh=0.15, frequency_thresh=0.6):
        if self.rounds.empty():  # no info = no confidence
            return {
                "opponent_next_throw": Throws['PAPER'],
                "frequency": 0,
                "consecutive_frequency": 0,
                "consecutive": 0,
                "confidence": 0
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

            generator = strat(computer=(not self.computer), rounds=Rounds(lookback_rounds))
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
        consecutive_frequency = consecutive / lookback
        g = strat(computer=(not self.computer), rounds=self.rounds)
        logging.info(f'{g.__class__.__name__} FREQUENCY: {frequency} CONSECUTIVE_FREQ: {consecutive_frequency}')
        confidence = frequency  # (consecutive_frequency + frequency)/2
        if consecutive > consecutive_freq_thresh:
            confidence *= 1.8  # reward if we have been more frequent than thresh
        if frequency < frequency_thresh:
            confidence *= 0.5  # penalize if hasn't been consistent
        confidence = min(confidence, 1)
        opponent_next_throw = g.throw()
        logging.info(f'OPPONENT NEXT THROW : {opponent_next_throw} CONFIDENCE: {confidence}')

        output = {
            "opponent_next_throw": opponent_next_throw,
            "frequency": frequency,
            "consecutive_frequency": consecutive_frequency,
            "consecutive": consecutive,
            "confidence": confidence
        }
        return output


class RandomStrat(Strategy):
    def throw(self):
        return random.choice(list(Throws)[0:3])  # random throw


class CopyStrat(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        if self.computer:
            return self.rounds.rounds[-1].p1  # i am p2, copy p1's last move
        return self.rounds.rounds[-1].p2  # i am p1, copy p2's last move


class BeatLastStrat(Strategy):
    def throw(self):
        if self.rounds.empty():
            return random.choice(list(Throws)[0:3])  # random throw
        if self.computer:
            return self.counter_throw(self.rounds.rounds[-1].p1)  # i am p2, beat p1's last move
        return self.counter_throw(self.rounds.rounds[-1].p2)  # i am p1, beat p2's last move


class CounterCopy(Strategy):
    def throw(self):
        result = self.counter_strat(CopyStrat, lookback=10, weight=1)
        if result["consecutive"] >= 3:
            return self.counter_throw(result["opponent_next_throw"])
        else:
            return random.choice(list(Throws)[0:3])  # random throw
