from uuid import uuid4

from flask import Blueprint, render_template, session


Index = Blueprint('Index', __name__)


@Index.get('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid4())
    return render_template('index.html')
