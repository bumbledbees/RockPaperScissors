# "Smart" Rock, Paper, Scissors

An attempt to build a rock, paper, scissors game that can perform better than
selecting moves at random versus a human opponent. Operates on a web-based
interface using Flask, and saves the results of its games to a SQLite database.


## Running the program

The web app uses Sessions to allow the server to uniquely identify users. For
this to work, you must create a `.env` file in the directory you're running the
program from, and set the SECRET\_KEY value as such:

```
SECRET_KEY=your-super-secret-key
```
The Flask documentation gives this command as an example of a quick way to
generate a secret key value:
```
$ python -c 'import secrets; print(secrets.token_hex())'
```

With python-poetry:
```
$ poetry run flask --app RockPaperScissors.app run
```

If you have the poetry-exec-plugin, you can simply run:
```
$ poetry exec app
```

Otherwise (ideally within a venv):
```
$ pip install -r requirements.txt
$ flask --app RockPaperScissors.app run
```
