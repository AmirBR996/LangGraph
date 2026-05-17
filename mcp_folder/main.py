from fastmcp import FastMCP
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "calories.db")

mcp = FastMCP("CalorieTracker")

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS calories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                food TEXT NOT NULL,
                calories REAL NOT NULL,
                protein REAL DEFAULT 0,
                note TEXT DEFAULT ''
            )
        """)

init_db()

@mcp.tool()
def add_calorie(date: str, food: str, calories: float, protein: float = 0, note: str = ""):
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            "INSERT INTO calories(date, food, calories, protein, note) VALUES (?, ?, ?, ?, ?)",
            (date, food, calories, protein, note)
        )
        return {"status": "ok", "id": cur.lastrowid}

@mcp.tool()
def list_calories(start_date: str, end_date: str):
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
            SELECT id, date, food, calories, protein, note
            FROM calories
            WHERE date BETWEEN ? AND ?
            ORDER BY date ASC
            """,
            (start_date, end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

@mcp.tool()
def daily_summary(date: str):
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
            SELECT date, SUM(calories), SUM(protein)
            FROM calories
            WHERE date = ?
            GROUP BY date
            """,
            (date,)
        )
        row = cur.fetchone()
        if not row:
            return {"date": date, "total_calories": 0, "total_protein": 0}
        return {"date": row[0], "total_calories": row[1], "total_protein": row[2]}

@mcp.tool()
def weekly_summary(start_date: str, end_date: str):
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
            SELECT date, SUM(calories) as total_calories
            FROM calories
            WHERE date BETWEEN ? AND ?
            GROUP BY date
            ORDER BY date ASC
            """,
            (start_date, end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

@mcp.resource("calorie://info", mime_type="application/json")
def info():
    return '{"tracker":"calorie","fields":["date","food","calories","protein"]}'

if __name__ == "__main__":
    mcp.run()