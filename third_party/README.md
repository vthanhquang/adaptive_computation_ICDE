# Third-party prompt sources (vendored verbatim, used only for baseline re-implementation)

| folder | source | licence |
|---|---|---|
| `din_sql/DIN-SQL_BIRD.py` | https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting (DIN-SQL, Pourreza & Rafiei, NeurIPS 2023) | see upstream repository |
| `mac_sql/const.py` | https://github.com/wbbeyourself/MAC-SQL (MAC-SQL, Wang et al., COLING 2025) | MIT |

Only the prompt-template string constants are read from these files (via `ast`), see
`plangate/classic_pipelines.py`. The files are never imported or executed.
