import functools
import os
import sqlite3


class Database(object):
    def __init__(self, db_path):
        db_path = os.path.abspath(db_path)
        # will create db if none found @ specified path

        self.conn = sqlite3.connect(
            db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False)

        self.cursor = self.conn.cursor()

        self.init_schema()

    def init_schema(self):
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_path, 'r') as schema:
            f = schema.read()
            self.conn.executescript(f)

    def execute(self, sql, params=()):
        with_params = f' with params {params}' if params else ''
        print(f'Executing {sql}{with_params}')
        self.cursor.execute(sql, params)
        self.conn.commit()

    @functools.lru_cache
    def columns(self, table: str):
        statement = f"SELECT name from PRAGMA_TABLE_INFO('{table}');"

        self.execute(statement)
        results = self.cursor.fetchall()
        return tuple([x[0] for x in results])

    def placeholders(self, data):
        return ', '.join('?' for _ in range(len(data)))

    def select_from(self, table: str, columns=(), predicate=None, sortby=None):
        col_names = ', '.join(columns) if columns else '*'
        constraint = f' WHERE {predicate}' if predicate else ''
        ordering = f' ORDER BY {sortby}' if sortby else ''
        statement = f'SELECT {col_names} FROM {table}{constraint}{ordering};'

        self.execute(statement)
        result = self.cursor.fetchall()

        columns = self.columns(table) if col_names == '*' else columns
        return {columns[i]: col for i, col in enumerate(zip(*result))}

    def insert_into(self, table: str, data: dict):
        columns = ', '.join(data.keys())
        placeholders = self.placeholders(data)
        statement = f'INSERT INTO {table}({columns}) VALUES({placeholders});'
        params = tuple(v for v in data.values())

        self.execute(statement, params)
        return self.cursor.fetchall()
