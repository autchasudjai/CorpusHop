<div align="center">

# CorpusHop

### Coverage-aware multi-hop question generation for corpus-aware RAG evaluation

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Evaluation-6C63FF)
![Multi-Hop QA](https://img.shields.io/badge/Multi--Hop-QA-0A7EA4)

**CorpusHop** automatically constructs verified three-hop question-answer pairs from a target corpus.  
It is designed to improve **corpus and topic coverage** when building evaluation sets for Retrieval-Augmented Generation (RAG).

</div>

---

## Overview

CorpusHop builds a directed graph over corpus chunks and repeatedly selects valid three-chunk reasoning paths with the **lowest degree-sum score**.

It supports two three-hop configurations.

### Bridge

```text
A → B → C
```

with chained triples:

```text
t1 = (s1, r1, s2)
t2 = (s2, r2, s3)
t3 = (s3,  r3, s4)
```

### Compositional

```text
A → C ← B
```

with:

```text
t1 = (s1, r1, s2)
t2 = (s4, r2, s2)
t3 = (s2, r3, s3)
```

Both `t1` and `t2` provide information required by `t3`.

---

## Pipeline

```text
Raw corpus / chunks
        │
        ▼
┌──────────────────────┐
│  Triple Extraction   │
│  extract_triples.py  │
└──────────┬───────────┘
           │
           ▼
  [Subject, Relation, Object]
           │
           ▼
┌──────────────────────┐
│ Directed Chunk Graph │
│ tail → head matching │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────┐
│ Greedy Path Selection      │
│ min Σ degree(v)            │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Generate 3 Sub-QA Pairs    │
│ + Compose Final Question   │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Validation + LLM-as-Judge  │
└──────────┬─────────────────┘
           │
           ▼
     Verified Multi-Hop QA
```

For candidate chunks `A`, `B`, and `C`:

```text
Score(A,B,C) = degree(A) + degree(B) + degree(C)
```

CorpusHop greedily selects the valid candidate with the **minimum degree-sum**.

If a generated example passes verification, all three chunks are removed from the live graph, producing **node-disjoint accepted paths**.

If generation or verification fails, only the two traversed edges are removed.

---

## Repository Structure

```text
CorpusHop/
│
├── Multi-hop questions/
│   ├── CorpusHop_CS/
│   │   ├── Bridge_Greedy.jsonl
│   │   ├── Bridge_Random1.jsonl
│   │   ├── Bridge_Random2.jsonl
│   │   ├── Bridge_Random3.jsonl
│   │   ├── Bridge_Random4.jsonl
│   │   ├── Bridge_Random5.jsonl
│   │   ├── Compositional_Greedy.jsonl
│   │   ├── Compositional_Random1.jsonl
│   │   ├── Compositional_Random2.jsonl
│   │   ├── Compositional_Random3.jsonl
│   │   ├── Compositional_Random4.jsonl
│   │   └── Compositional_Random5.jsonl
│   │
│   ├── CorpusHop_Leaflets/
│   │   ├── Bridge_Greedy.jsonl
│   │   ├── Bridge_Random1.jsonl
│   │   ├── Bridge_Random2.jsonl
│   │   ├── Bridge_Random3.jsonl
│   │   ├── Bridge_Random4.jsonl
│   │   ├── Bridge_Random5.jsonl
│   │   ├── Compositional_Greedy.jsonl
│   │   ├── Compositional_Random1.jsonl
│   │   ├── Compositional_Random2.jsonl
│   │   ├── Compositional_Random3.jsonl
│   │   ├── Compositional_Random4.jsonl
│   │   └── Compositional_Random5.jsonl
│   │
│   └── CorpusHop_MuSiQue/
│       ├── Bridge_greedy.jsonl
│       └── Compositional_greedy.jsonl
│
├── Bridge.ipynb
├── Compositional.ipynb
├── example_chunks.jsonl
├── extract_triples.py
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/autchasudjai/CorpusHop.git
cd CorpusHop
```

### 2. Install dependencies

```bash
pip install openai python-dotenv numpy scipy networkx jupyter
```

### 3. Set your OpenAI API key

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your_api_key_here
```

> Do not commit `.env` to GitHub.

---

## Step 1 — Extract Triples

A small example corpus is included:

```text
example_chunks.jsonl
```

Run:

```bash
python extract_triples.py --input example_chunks.jsonl --output example_chunks_with_triples.jsonl
```

The extractor supports:

- `.json`
- `.jsonl`
- `.ndjson`
- `.txt`
- a single input file
- a directory containing multiple input files

It automatically detects common text fields such as:

```text
text
content
context
passage
chunk
chunk_text
document
body
```

and common ID fields such as:

```text
id
chunk_id
doc_id
document_id
passage_id
node_id
```

### Minimal Input

```json
{"id": "doc-001", "topic": "science", "text": "Marie Curie discovered polonium."}
```

### Extracted Output

```json
{
  "id": "doc-001",
  "topic": "science",
  "text": "Marie Curie discovered polonium.",
  "extracts": [
    ["marie curie", "discovered", "polonium"],
    ["polonium", "was discovered by", "marie curie"]
  ],
  "entities": [
    "marie curie",
    "polonium"
  ]
}
```

The output JSONL can be used directly by both generation notebooks.

---

## Step 2 — Generate Bridge Questions

Open:

```text
Bridge.ipynb
```

Point the configuration to the extracted data:

```python
INPUT_DATA = Path("./example_chunks_with_triples.jsonl")
OUTPUT_DIR = Path("./outputs/bridge")

RUN_CLEANING = True
RUN_GRAPH_BUILDING = True
RUN_LLM_PIPELINE = True
```

Bridge searches for:

```text
A → B → C
```

and constructs:

```text
t1 = (s1, r1, s2)
t2 = (s2, r2, X)
t3 = (X,  r3, s3)
```

---

## Step 3 — Generate Compositional Questions

Open:

```text
Compositional.ipynb
```

Use the same extracted dataset:

```python
INPUT_DATA = Path("./example_chunks_with_triples.jsonl")
OUTPUT_DIR = Path("./outputs/compositional")

RUN_CLEANING = True
RUN_GRAPH_BUILDING = True
RUN_LLM_PIPELINE = True
```

Compositional searches for:

```text
A → C ← B
```

and constructs:

```text
t1 = (s1, r1, s2)
t2 = (s4, r2, s2)
t3 = (s2, r3, s3)
```

---

## Greedy Selection

For every valid candidate:

```text
score = degree(A) + degree(B) + degree(C)
```

The candidate with the **minimum degree-sum** is selected.

The live graph is updated after every attempt:

```text
PASS
└── save the verified QA pair
    └── remove A, B, and C

FAIL
└── remove only the two traversed edges
```

Selection continues until no valid candidate remains.

---

## Question Validation

For each selected triple chain, the pipeline generates three sub-question-answer pairs.

The sub-question checks enforce the intended hop dependencies and prevent answer leakage.

For Bridge:

```text
Q1 → answer s2
Q2 uses s2 → answer X
Q3 uses X  → answer s3
```

The expected answer entity must not already appear in its corresponding sub-question.

After composing the final multi-hop question, an **LLM-as-judge** verifies that:

1. the final question requires the intended three-hop reasoning chain, and
2. the final answer correctly answers the generated question.

---

## Generated Outputs

Example output layout:

```text
Multi-hop questions/
└── CorpusHop_CS/
    ├── Bridge_Greedy.jsonl
    ├── Bridge_Random1.jsonl
    ├── ...
    ├── Compositional_Greedy.jsonl
    └── Compositional_Random5.jsonl
```

Random selection is repeated across multiple runs so its variation can be compared with greedy path selection.

---

## Included Corpora

Generated outputs are organized for:

- **CS** — computer science handbook corpus
- **Leaflets** — medical leaflet corpus
- **MuSiQue** — multi-hop QA corpus

---

## Reproducibility Notes

- Triple links use normalized entity strings.
- Chunk-graph edges follow the required dependency direction.
- Greedy selection uses the lowest degree-sum candidate.
- Accepted paths are node-disjoint.
- Graph degrees are recomputed as the live graph changes.
- Generalized scripts avoid machine-specific absolute paths.
- LLM execution is disabled by default in the notebooks to avoid accidental API usage.
- Random runs are stored separately.

---

## Files

| File | Purpose |
|---|---|
| `extract_triples.py` | Extract and clean subject-relation-object triples |
| `Bridge.ipynb` | Generate Bridge-style three-hop QA pairs |
| `Compositional.ipynb` | Generate Compositional-style three-hop QA pairs |
| `example_chunks.jsonl` | Small example corpus for testing |
| `Multi-hop questions/` | Generated greedy and random QA sets |

---

## Research

**CorpusHop: Multi-Hop Question Generation for Corpus-Aware RAG Evaluation**

Authors:

- Autcha Sudjai
- Pat Vatiwutipong
- Wasin Faengrit
- Thanapon Noraset

Citation information can be added once the official proceedings entry is available.

---

<div align="center">

### Corpus coverage first. Multi-hop reasoning by construction.

</div>
