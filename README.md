# GraphHop: Hybrid Graph + Vector RAG with LangGraph

Multi-hop question answering over documents. A **knowledge graph (Neo4j)** and a **vector index (pgvector)** share the same chunk IDs, and a **LangGraph state machine** routes each question, retrieves, answers with citations, and validates them.

> Plain vector RAG answers "Who is the CEO of Norda?" well. It fails on "Who is the CEO of the company that acquired the supplier of the partner of X?", because no single chunk contains the whole chain. GraphHop walks the chain in the graph instead.

## Results

Two evaluation sets, reported separately. Same model (`openai/gpt-oss-120b` on Groq), temperature 0.

### 1. Synthetic benchmark (56 questions, 1 to 4 hops)

| System | 1-hop | 2-hop | 3-hop | 4-hop | agg | oos | Overall | p50 ms | p95 ms | tokens/query | LLM calls/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vector-only baseline | 100% (n=10) | 25% (n=12) | 0% (n=12) | 0% (n=10) | 100% (n=6) | 100% (n=6) | 45% | 840 | 1902 | 877 | 1.0 |
| **Hybrid graph+vector** | 100% (n=10) | 100% (n=12) | 100% (n=12) | 100% (n=10) | 100% (n=6) | 100% (n=6) | **100%** | 2347 | 5049 | 2,254 | 2.7 |

### 2. Hand-written set over a small real-style corpus (10 questions)

| System | 1-hop | 2-hop | 3-hop | agg | oos | Overall | p50 ms | p95 ms | tokens/query | LLM calls/query |
|---|---|---|---|---|---|---|---|---|---|---|
| Vector-only baseline | 100% (n=2) | 100% (n=2) | 33% (n=3) | 100% (n=1) | 100% (n=2) | 80% | 750 | 1605 | 869 | 1.0 |
| **Hybrid graph+vector** | 100% (n=2) | 100% (n=2) | 100% (n=3) | 100% (n=1) | 100% (n=2) | **100%** | 2236 | 4201 | 1,807 | 2.6 |

### Cost of the improvement

| Metric | Vector-only | Hybrid | Ratio |
|---|---|---|---|
| p50 latency (synthetic) | 840 ms | 2347 ms | about 2.8x |
| Tokens per query (synthetic) | 877 | 2,254 | about 2.6x |
| LLM calls per query | 1.0 | 2.7 | 2.7x |

Hybrid costs more per query. It pays off when questions need more than one hop: 0% to 100% on 3-hop and 4-hop questions.

### Honest caveats

- The synthetic set is generated from templates, and the graph is built from the same generator. It shows the mechanism clearly, but it is easier than real text.
- The second set has only 10 questions over 4 short documents. It is a sanity check on non-templated text, not a benchmark. Larger real-data evaluation is future work.
- Out-of-scope ("oos") questions must be refused. Both systems refuse them correctly in both sets.

## Architecture

```
START -> route -> graph  ----+
              \-> vector ----+--> merge -> generate -> validate --(citations invalid)--> generate (retry, max 3)
                                                          \--(valid or refused)--> END
```

| Node | What it does |
|---|---|
| route | A small LLM call picks `graph`, `vector` or `both`. Low confidence means `both`, run in parallel. |
| graph | Template-based Cypher retrieval. The model never writes Cypher. A plan like `PARTNERS_WITH -> SUPPLIES -> ACQUIRED -> CEO_OF` is derived from the question and walked in Neo4j. |
| vector | pgvector HNSW search with local sentence-transformers embeddings. Also the fallback when the graph returns fewer than 3 facts. |
| merge | Labels graph facts and passages, and makes every cited chunk ID resolve to real text. |
| generate | LangChain prompt to ChatOpenAI. Answers with chunk-ID citations. |
| validate | Every citation must be a retrieved chunk ID. If not, the error is fed back and the answer regenerated (max 3 attempts). |

### Design decisions

| Decision | Why |
|---|---|
| Shared chunk IDs across graph and vector stores | Every graph fact is traceable to the exact source text. |
| Template Cypher, not LLM-written Cypher | Deterministic and safe. No injection or hallucinated queries. |
| Entity resolution (`Acme`, `Acme Inc.`, `ACME CORPORATION`) | Chains do not break on name variants. |
| Citation validation loop | Prevents fabricated sources. |
| Deterministic rescue for complete chains | See below. |
| Own retry loop for rate limits, with waits excluded from latency | Honest latency numbers on a free-tier API. |

### Reliability: complete-chain rescue

`gpt-oss-120b` on Groq is not fully deterministic even at temperature 0. On some 4-hop questions it answered `INSUFFICIENT_CONTEXT` although the full chain was in the prompt. The fix, in `app/pipeline.py`:

1. If the graph returned a complete chain (as many facts as plan steps) and the model refuses, re-ask with a focused prompt containing only the ordered hops and their cited passages (up to 2 rescues).
2. If it still refuses, answer from the last verified graph fact.

Incomplete chains still refuse correctly, so out-of-scope behaviour is unchanged (oos stays at 100%). Effect: 4-hop accuracy went from 80% to 100%, and the two previously flaky questions passed 5/5 repeated runs.

## Project structure

```
app/
  answer.py      public entry point: ask(question, mode)
  pipeline.py    LangGraph state machine (route, graph, vector, merge, generate, validate)
  router.py      graph / vector / both routing and decision log
  graph.py       template Cypher retrieval and hop planning
  vector.py      pgvector search and chunk lookup
  llm.py         ChatOpenAI client, rate-limit handling, usage counters, embeddings
  ingest.py      documents -> chunks -> graph + vectors
  extract.py     relation extraction
  resolve.py     entity resolution
  chunking.py    document chunking
  schema.py      relationship types
eval/
  run_eval.py             benchmark runner (resumable)
  questions_synthetic.json
  questions.json
  vector_recall.py
data/
  synthetic/     generated corpus
  docs/          small real-style corpus
results/         benchmark tables
docs/            HOW_IT_WORKS.md, pipeline.mmd
```

## Setup

Requirements: Python 3.10+, Neo4j, PostgreSQL with pgvector, and an OpenAI-compatible LLM endpoint (Groq, Gemini, Ollama, ...).

```bash
git clone https://github.com/ezhilarasi0509/graphhop_langgraph.git
cd graphhop_langgraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Configuration is read in `app/config.py`. Set at least the LLM endpoint, key and model (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`), plus your Neo4j and Postgres connection settings. **Never commit keys.**

### Ingest data

```bash
# estimate cost first
DATA_DIR=data/synthetic python -m app.ingest --dry-run

# synthetic corpus
DATA_DIR=data/synthetic python -m app.ingest

# small real-style corpus
DATA_DIR=data/docs python -m app.ingest
```

Ingestion is idempotent (`MERGE`, cached LLM calls). Inspect the graph at http://localhost:7474.

### Ask a question

```python
from app import answer
r = answer.ask("Who is the CEO of the company that acquired the supplier of the partner of Northern Ridge?", "auto")
print(r["answer"], r["graph_facts"], r["citations"])
```

The result includes the answer, the route taken, graph facts, resolved citations, latency, tokens and LLM calls.

### Reproduce the benchmarks

```bash
python -m eval.run_eval --questions eval/questions_synthetic.json --out results/benchmark_synthetic.md
python -m eval.run_eval --questions eval/questions.json --out results/benchmark_real.md
```

Runs resume from `results/progress.jsonl`. Delete a row to recompute that question.

## Limitations and future work

- Free-tier rate limits (8,000 tokens/min on Groq) slow ingestion and evaluation.
- Larger real-data evaluation with independent annotators.
- Hop planning currently uses fixed relationship templates.
- Cost: hybrid uses about 2.7x the LLM calls of vector-only.

## Built with

LangGraph, LangChain, Neo4j, pgvector, sentence-transformers, Groq (`gpt-oss-120b`).

Built for the **No Hands: Wispr Challenge** (WeMakeDevs x Wispr Flow), with the README, prompts and commit messages dictated by voice using Wispr Flow.
