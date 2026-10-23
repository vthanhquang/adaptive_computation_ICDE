"""Build a k-times scaled copy of the BIRD PostgreSQL database in schema x<k>.

Every table is replicated k times. In copy i, every key column (primary key, foreign key, referenced
or uniquely indexed column) is shifted: integers by i*10^9, text by suffix '#i'; copy 0 is the
original. Joins therefore stay within a copy, so row counts and join work grow linearly in k and the
optimizer sees realistic statistics, while non-key values (and hence literal predicates) are
preserved. Query RESULTS on the scaled schema are NOT the original answers (counts and aggregates
change), so this schema is used for resource measurements only, never for accuracy.

Usage: python -m experiments.scaling.build_scaled_database 10    (and 100; about 10 GB and 101 GB)
"""
import sys, time

import psycopg

from plangate.engines.postgres import DSN

INT_TYPES = {"integer", "bigint", "smallint"}
NUM_TYPES = {"real", "double precision", "numeric"}
TXT_TYPES = {"text", "character varying", "character"}


def main():
    k = int(sys.argv[1])
    sch = f"x{k}"
    c = psycopg.connect(DSN, autocommit=True)
    c.execute("SET statement_timeout = 0")
    c.execute(f"DROP SCHEMA IF EXISTS {sch} CASCADE"); c.execute(f"CREATE SCHEMA {sch}")
    tables = [r[0] for r in c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' "
                                      "AND table_type='BASE TABLE' ORDER BY 1")]
    keycols = set(c.execute("""
        SELECT kcu.table_name, kcu.column_name FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema='public' AND tc.constraint_type IN ('PRIMARY KEY','FOREIGN KEY')
        UNION
        SELECT ccu.table_name, ccu.column_name FROM information_schema.table_constraints tc
        JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_schema='public' AND tc.constraint_type='FOREIGN KEY'""").fetchall())
    idx = c.execute("SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public'").fetchall()
    for t, _, d in idx:
        if "UNIQUE" in d:
            cols = d[d.rindex("(") + 1: d.rindex(")")].split(",")
            keycols |= {(t, x.strip().strip('"')) for x in cols}
    t0 = time.time()
    for t in tables:
        cols = c.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' "
                         "AND table_name=%s ORDER BY ordinal_position", (t,)).fetchall()
        sel = []
        for name, ty in cols:
            q = '"' + name.replace('"', '""') + '"'
            if (t, name) in keycols:
                if ty in INT_TYPES:
                    sel.append(f"(s.{q}::bigint + g.i::bigint * 1000000000) AS {q}")
                elif ty in NUM_TYPES:
                    sel.append(f"(s.{q} + g.i::bigint * 1000000000) AS {q}")
                elif ty in TXT_TYPES:
                    sel.append(f"(CASE WHEN g.i = 0 THEN s.{q} ELSE s.{q} || '#' || g.i END) AS {q}")
                else:
                    sel.append(f"s.{q}")
            else:
                sel.append(f"s.{q}")
        c.execute(f'CREATE TABLE {sch}."{t}" AS SELECT {", ".join(sel)} FROM public."{t}" s '
                  f'CROSS JOIN generate_series(0, {k - 1}) AS g(i)')
        print(f"  {t}: {time.time()-t0:.0f}s", flush=True)
    for t, name, d in idx:
        d2 = d.replace(" ON public.", f" ON {sch}.").replace(f"INDEX {name}", f"INDEX {name}_{sch}")
        try:
            c.execute(d2)
        except Exception as e:   # a shifted key could still collide (non-int/text key): keep a non-unique index
            c.execute(d2.replace("UNIQUE ", ""))
            print("  non-unique:", name, str(e)[:80])
    c.execute("ANALYZE")
    size = c.execute(f"SELECT pg_size_pretty(sum(pg_total_relation_size(format('%I.%I', table_schema, table_name)))) "
                     f"FROM information_schema.tables WHERE table_schema='{sch}'").fetchone()[0]
    print(f"schema {sch}: {len(tables)} tables, {size}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
