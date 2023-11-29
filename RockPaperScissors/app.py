import os

from dotenv import dotenv_values
from flask import Flask

from RockPaperScissors.blueprints.index import Index
from RockPaperScissors.blueprints.api import API
from RockPaperScissors.db import Database

basedir = os.path.abspath(os.getcwd())

app = Flask(__name__)
app.config.update(dotenv_values(os.path.join(basedir, '.env')))
app.config.update(SESSION_COOKIE_SAMESITE='Strict')

if app.config.get('SECRET_KEY', None) is None:
    raise Exception('Error: secrets not provided in .env file')
app.secret_key = app.config['SECRET_KEY']

app.database = Database('rps.db')

app.register_blueprint(Index)
app.register_blueprint(API)
