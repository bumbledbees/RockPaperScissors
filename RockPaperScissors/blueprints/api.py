from datetime import datetime
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, session

from RockPaperScissors import Game, Strategies


API = Blueprint('API', __name__)

STRATEGY = Strategies.SameliaBot


@API.route('/api/move', methods=('GET', 'POST'))
def move():
    if request.is_json:
        body = request.json
    else:
        raise Exception('request is not json')

    if 'user_id' not in session:
        session['user_id'] = str(uuid4())
    user_id = session['user_id']

    if 'playerMove' not in body:
        raise Exception(f'{body}')

    player_move = Game.Throws(body['playerMove'])
    if player_move not in Game.Throws:
        raise Exception(f'invalid move: {player_move}')

    history = current_app.database.select_from(
        'rps', ('player_move', 'computer_move'),
        predicate=f"user_id = '{user_id}'",
        sortby='time')
    history = zip(history['player_move'], history['computer_move'])
    history = map(lambda p1_p2: tuple(Game.validate_throw(p) for p in p1_p2),
                  history)

    computer_move = STRATEGY(Game.Rounds([Game.Round(*r) for r in history])).throw()

    winner = Game.evaluate_game(player_move, computer_move)

    current_app.database.insert_into('rps', {
        'user_id': user_id,
        'time': datetime.now(),
        'player_move': player_move[0],
        'computer_move': computer_move[0],
        'winner': winner[0]
    })

    return jsonify({'computerMove': computer_move, 'winner': winner})
