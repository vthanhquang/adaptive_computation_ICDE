"""Few-shot exemplars for the few-shot baseline prompts (AutoMix verification, CHASE-SQL selection).

Questions are BIRD dev items that are in NEITHER evaluation set (300 SQLite sample, 500 mini-dev),
so no evaluation question leaks into a prompt. Correct SQL = BIRD gold; incorrect SQL = a
hand-written variant with an error type observed in the error analysis (value-case mismatch,
wrong column, wrong aggregate). Execution results are computed from the SQLite databases.
"""
from __future__ import annotations
from functools import lru_cache

from ..engines.sqlite import connect, run_bounded

EXEMPLARS = [
    {"question_id": 853, "db_id": "formula_1",
     "question": "Please give the names of the races held on the circuits in Spain.",
     "evidence": "Spain is a name of country;",
     "correct": "SELECT DISTINCT T2.name FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.country = 'Spain'",
     "wrong": "SELECT DISTINCT T2.name FROM circuits AS T1 INNER JOIN races AS T2 ON T2.circuitID = T1.circuitId WHERE T1.country = 'spain'",
     "why_wrong": "The filter value 'spain' does not match the stored value 'Spain', so the query returns no rows."},
    {"question_id": 404, "db_id": "card_games",
     "question": "Indicates the name of all the languages into which the set whose number of cards is 309 is translated.",
     "evidence": "set refer to setCode; number of cards refers to baseSetSize; baseSetsize = 309",
     "correct": "SELECT T2.language FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.baseSetSize = 309",
     "wrong": "SELECT T1.name FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T1.baseSetSize = 309",
     "why_wrong": "The question asks for the languages of the translations, but the query returns the set names."},
    {"question_id": 1510, "db_id": "debit_card_specializing",
     "question": "What is the average total price of the transactions taken place in gas stations in the Czech Republic?",
     "evidence": "Gas station in the Czech Republic implies that Country = 'CZE'",
     "correct": "SELECT AVG(T1.Price) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T2.Country = 'CZE'",
     "wrong": "SELECT SUM(T1.Price) FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID WHERE T2.Country = 'CZE'",
     "why_wrong": "The question asks for the average price, but the query computes the SUM."},
]


@lru_cache(maxsize=None)
def result_text(db_id, sql):
    conn = connect(db_id)
    rows, err, _, _ = run_bounded(conn, sql, timeout_s=10)
    conn.close()
    if err:
        return f"execution error ({err[:100]})"
    return f"{len(rows)} rows; first rows: {[list(map(lambda v: str(v)[:30], r)) for r in rows[:3]]}"
