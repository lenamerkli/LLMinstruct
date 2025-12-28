import sqlite3
import os
import atexit

_db_connection = None

def get_db() -> sqlite3.Connection:
    global _db_connection
    if _db_connection is None:
        _db_connection = sqlite3.connect('database.sqlite3')
    return _db_connection


def close_connection() -> None:
    global _db_connection
    if _db_connection is not None:
        _db_connection.close()  # type: ignore
        _db_connection = None


def query_db(query, args=(), one=False) -> list | tuple:
    conn = get_db()
    cur = conn.execute(query, args)
    result = cur.fetchall()
    conn.commit()
    cur.close()
    return (result[0] if result else None) if one else result


def relative_path(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)


# Initialize the database
def init_db():
    _create_db = """
    CREATE TABLE IF NOT EXISTS messages (
        prompt TEXT,
        response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS translations (
        original TEXT,
        translation TEXT,
        language TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = get_db()
    conn.executescript(_create_db)
    conn.commit()

atexit.register(close_connection)

init_db()

if __name__ == '__main__':
    pass
