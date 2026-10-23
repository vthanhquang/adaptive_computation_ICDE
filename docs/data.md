# Obtaining the large files (not stored in git)

The repository holds code, results, figures and reports. Four kinds of large or
machine-specific data are excluded by `.gitignore`. This file explains how to obtain or
rebuild each of them.

| Path | Size | What | Needed for |
|---|---|---|---|
| `data/bird/` | ~1.4 GB | BIRD dev set + 11 SQLite databases | every SQLite experiment |
| `data/pg/minidev/` | ~0.95 GB | BIRD mini-dev (PostgreSQL questions + dump) | PostgreSQL arm |
| `data/pg/pgsql/`, `data/pg/cluster/` | ~2.9 GB | PostgreSQL 16 binaries and the loaded database cluster | PostgreSQL arm |
| `cache/` | ~15 MB (shipped as `artifacts/cache.zip`) | cached LLM responses and DB evidence | reproducing results **without** API calls |
| HuggingFace cache | ~2 GB | EllieSQL router checkpoint | EllieSQL baseline only |

After the steps below, the layout should be:

```
<repository root>/
  data/bird/dev_20240627/{dev.json, dev_tables.json, dev_tied_append.json, dev_databases/<db_id>/<db_id>.sqlite}
  data/pg/minidev/MINIDEV/{mini_dev_postgresql.json, ...}
  data/pg/minidev/MINIDEV_postgresql/BIRD_dev.sql
  data/pg/pgsql/bin/{postgres, pg_ctl, initdb, psql}      # or any PostgreSQL 16 on PATH
  data/pg/cluster/                                        # created by initdb
  cache/{llm, evidence, pg_evidence}/*.json
```

---

## 1. Python environment

```bash
pip install -r requirements.txt
```

Python 3.11+ is recommended; the experiments were run with Python 3.13 on Windows 11. The
EllieSQL baseline also needs `torch` and `transformers` (CPU is enough), which are included
in `requirements.txt`.

## 2. BIRD dev (SQLite arm), ~1.4 GB

Source: BIRD benchmark, https://bird-bench.github.io/ (dev set, release 2024-06-27).

**Linux / macOS**

```bash
mkdir -p data/bird && cd data/bird
curl -L -o dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip                                   # -> dev_20240627/
cd dev_20240627 && unzip dev_databases.zip && rm dev_databases.zip
cd ../ && rm dev.zip
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force data\bird | Out-Null; Set-Location data\bird
curl.exe -L -o dev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
Expand-Archive dev.zip -DestinationPath . -Force
Set-Location dev_20240627; Expand-Archive dev_databases.zip -DestinationPath . -Force
Remove-Item dev_databases.zip, ..\dev.zip
```

Check: `data/bird/dev_20240627/dev.json` has 1534 questions, and
`dev_databases/california_schools/california_schools.sqlite` exists. The scorer
(`plangate/bird.py`) also reads the official tied-answer alternatives from
`dev_tied_append.json`, which ships inside `dev.zip`.

## 3. BIRD mini-dev (PostgreSQL arm), ~0.8 GB download

Source: BIRD mini-dev, https://github.com/bird-bench/mini_dev

```bash
mkdir -p data/pg && cd data/pg
curl -L -C - -o minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
python - <<'EOF'
import zipfile
z = zipfile.ZipFile("minidev.zip")
for m in ["minidev/MINIDEV_postgresql/BIRD_dev.sql", "minidev/MINIDEV/mini_dev_postgresql.json",
          "minidev/MINIDEV/mini_dev_sqlite.json", "minidev/MINIDEV/mini_dev_postgresql_gold.sql"]:
    z.extract(m, ".")
EOF
```

The download is about 800 MB. If it stops early, re-run the same `curl -C -` command to resume
it, and check that the file is 800,943,648 bytes before extracting.

## 4. PostgreSQL 16 with the BIRD dump

Any PostgreSQL 16 server works. The code connects to
`host=localhost port=5544 user=postgres dbname=bird` (see `DSN` in `plangate/pg.py`). Change
it there if you use another port or user.

**Option A: portable binaries (Windows, no admin rights), as used in the experiments**

```powershell
Set-Location data\pg
curl.exe -L -o pg16.zip https://get.enterprisedb.com/postgresql/postgresql-16.4-1-windows-x64-binaries.zip
Expand-Archive pg16.zip -DestinationPath . -Force          # -> data\pg\pgsql\
$pg = (Resolve-Path .\pgsql\bin).Path
& "$pg\initdb.exe" -D cluster -U postgres -A trust -E UTF8 --locale=C
& "$pg\pg_ctl.exe" -D cluster -l server.log -o "-p 5544 -c listen_addresses=localhost -c shared_buffers=1GB -c work_mem=64MB" start
```

**Option B: system PostgreSQL 16 (Linux / macOS)**

```bash
# e.g. apt install postgresql-16   or   brew install postgresql@16
initdb -D data/pg/cluster -U postgres -A trust -E UTF8 --locale=C
pg_ctl -D data/pg/cluster -l data/pg/server.log -o "-p 5544 -c listen_addresses=localhost" start
```

**Load the dump (both options)**

```bash
psql -h localhost -p 5544 -U postgres -c "CREATE ROLE xiaolongli;" -c "CREATE DATABASE bird;"
psql -h localhost -p 5544 -U postgres -d bird -q -f data/pg/minidev/MINIDEV_postgresql/BIRD_dev.sql
psql -h localhost -p 5544 -U postgres -d bird -c "ANALYZE;"
```

The dump refers to the owner role `xiaolongli`, so create that role first. Loading takes
about 30 s and yields 75 tables (about 1 GB). Check:

```bash
psql -h localhost -p 5544 -U postgres -d bird -c "select count(*) from pg_tables where schemaname='public';"   # 75
```

The server must be running for every PostgreSQL experiment. Start it again after a reboot with
the `pg_ctl ... start` command above.

Security note: the cluster uses `trust` authentication and listens only on `localhost`.
`plangate/pg.py` additionally opens every session read-only and sets a statement timeout.

## 5. Cached LLM responses and DB evidence (`artifacts/cache.zip`)

All LLM calls and DB evidence used in the reported results are cached. Unpacking the archive
reproduces every table and figure **without** any API calls or API cost:

```bash
python -c "import zipfile; zipfile.ZipFile('artifacts/cache.zip').extractall('.')"
```

This creates `cache/llm/` (one JSON file per (model, messages, temperature, sample) key),
`cache/evidence/` (SQLite evidence) and `cache/pg_evidence/` (PostgreSQL evidence).

To run **new** generations instead, set an API key for each provider whose models you call.
`plangate/providers/official.py` sends every model to its provider's official
OpenAI-compatible endpoint:

| Models | Provider | Environment variable |
|---|---|---|
| `deepseek-v4-flash`, `deepseek-v4.1-flash`, `deepseek-v4-pro` | DeepSeek | `DEEPSEEK_API_KEY` |
| `glm-5.3-flash` | Z.ai | `ZAI_API_KEY` |
| `kimi-k3` | Moonshot AI | `MOONSHOT_API_KEY` |
| `qwen3.8-max` | Alibaba Cloud (DashScope) | `DASHSCOPE_API_KEY` |
| `hy3` | Tencent Hunyuan | `HUNYUAN_API_KEY` |

Each base URL can be overridden with `<PROVIDER>_BASE_URL` (e.g. `DEEPSEEK_BASE_URL`). Keys
are never stored in the repository or in the cache. Calls that miss the cache are billed. The
full set of experiments made roughly 25k calls. Token counts, and therefore costs, are the ones
recorded in the cache; a fresh run can report slightly different token counts.

## 6. EllieSQL router checkpoint (only for the EllieSQL baseline)

`experiments/run_elliesql.py` and `experiments/retrain_ellie_router.py` download the released
router `derrickzhu/EllieSQL_Router_Checkpoints` (subfolder `qwen_dpo`, about 2 GB) from the
HuggingFace Hub automatically on first use, into `~/.cache/huggingface/hub`. It runs on CPU.
For offline runs after the first download, set `HF_HUB_OFFLINE=1`.

## 7. Quick check that everything is in place

```bash
python -c "import sys; sys.path.insert(0,'.'); from plangate.bird import load_dev; print(len(load_dev()))"          # 1534
python -c "import sys; sys.path.insert(0,'.'); from plangate import pg; print(len(pg.load_minidev()))"            # 500 (needs step 3)
python experiments/summary_table.py      # rebuilds the summary table from results/*.json (no data needed)
python experiments/planner_d.py sqlite   # re-runs Experiment D from the cache (needs steps 2 and 5)
```
