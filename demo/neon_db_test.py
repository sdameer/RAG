# create data 
import os
import psycopg
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

conn_string = os.getenv("DATABASE_URL")

def create_table():
    with psycopg.connect(conn_string) as conn:  # type: ignore
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    answer TEXT
                );
            """)
            
            
def save_chat(query, answer):
    with psycopg.connect(conn_string) as conn:  # type: ignore
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (query, answer)
                VALUES (%s, %s);
                """,
                (query, answer)
            )
            
def print_chat_history():
    with psycopg.connect(conn_string) as conn:  # type: ignore
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, query, answer
                FROM chat_history
                ORDER BY id;
            """)

            rows = cur.fetchall()

            print("\n--- Stored Chat History ---")

            for row in rows:
                print(f"ID: {row[0]}")
                print(f"Query: {row[1]}")
                print(f"Answer: {row[2]}")
                print()
                        
                        
def previous_chat_history() :
    with psycopg.connect(conn_string) as conn:  # type: ignore
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, query, answer
                FROM chat_history
                ORDER BY id;
            """)

            rows = cur.fetchall()
            chat_history = []
            for row in rows:
                chat_history.append(HumanMessage(content = row[1]))
                chat_history.append(AIMessage(content=row[2]))

    return chat_history
                                         
