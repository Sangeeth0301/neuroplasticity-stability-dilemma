import sqlite3
import pandas as pd

def create_database():
    conn = sqlite3.connect('q_contextsync.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_log (
            turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            speaker TEXT,
            intent TEXT,
            raw_text TEXT
        )
    ''')
    
    # The Synthetic 5-Member TCL-BC Benchmark
    chat_data = [
        ("Member 1 (Architect)", "Rule Setting", "We are building a Quantum LLM. Rule 1: Always use amplitude encoding. Rule 2: Never use dense matrix inversions. Rule 3: The output must always be JSON."),
        ("Member 2 (Expander)", "Feature Addition", "Understood. How do we write the parameter-shift rule to calculate the S-FIM for the quantum gates?"),
        ("Member 3 (Data Dumper)", "Context Saturation", "{'data': 'massive_payload', 'tokens': [45, 99, 102], 'status': 'injection', 'payload': 'overflow_context_window'}"),
        ("Member 4 (Domain Shifter)", "Adversarial Noise", "Hey everyone, completely off topic, but does anyone have a good recipe for chocolate chip cookies? I need to bake some tonight."),
        ("Member 5 (Tester)", "Final Evaluation", "Before we wrap up, can someone summarize the project rules we agreed on at the very beginning of the thread?")
    ]
    
    cursor.executemany('''
        INSERT INTO chat_log (speaker, intent, raw_text) 
        VALUES (?, ?, ?)
    ''', chat_data)
    
    conn.commit()
    print("Database created and 5-Node Synthetic Chat Log saved successfully.")
    conn.close()

if __name__ == "__main__":
    create_database()
    