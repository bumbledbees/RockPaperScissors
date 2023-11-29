BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS rps (
    id INTEGER PRIMARY KEY ASC AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    time INTEGER NOT NULL,
    player_move TEXT CHECK(player_move IN ('R', 'P', 'S')) NOT NULL,
    computer_move TEXT CHECK(computer_move IN ('R', 'P', 'S')) NOT NULL,
    winner TEXT CHECK(winner IN ('P', 'C', 'T')) NOT NULL
);

COMMIT TRANSACTION;
