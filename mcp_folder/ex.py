import sqlite3

conn = sqlite3.connect("calories.db")
cur = conn.execute("SELECT * FROM calories")
print(cur.fetchall())