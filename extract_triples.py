"""
General-purpose triple extraction for CorpusHop.

Input:
- A single .json / .jsonl / .ndjson / .txt file, or
- A directory containing those files.

Output:
- JSONL in the canonical CorpusHop schema:
    {
        "id": ...,
        "topic": ...,
        "text": "...",
        "extracts": [["subject", "relation", "object"], ...],
        "entities": ["entity1", "entity2", ...]
    }

Typical usage:
    python extract_triples_generalized.py \
        --input ./data/chunks.jsonl \
        --output ./data/chunks_with_triples.jsonl

Environment:
    OPENAI_API_KEY=...
Optional:
    OPENAI_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = 0.0

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".txt"}

# Set any value manually if your dataset uses unusual field names.
# Example:
# FIELD_MAP = {
#     "id": "chunk_key",
#     "text": "passage",
#     "topic": "category",
# }
FIELD_MAP: dict[str, str | None] = {
    "id": None,
    "text": None,
    "topic": None,
}

FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "id": (
        "id",
        "chunk_id",
        "chunkid",
        "doc_id",
        "document_id",
        "passage_id",
        "node_id",
        "key",
    ),
    "text": (
        "text",
        "content",
        "context",
        "passage",
        "chunk",
        "chunk_text",
        "document",
        "body",
    ),
    "topic": (
        "topic",
        "category",
        "label",
        "section",
        "title",
        "source",
        "document_title",
    ),
}


SYSTEM_PROMPT = """You are a knowledge-graph extraction model. Extract factual triplets from text.

OUTPUT FORMAT:
Return a valid JSON array of [Subject, Relation, Object] arrays only.
Do not use markdown and do not add explanations.

STRICT RULES:
1. Use ATOMIC entities only — short noun phrases (prefer 1-5 words).
   BAD:  "we cannot ask a man what he will do and if we should..."
   GOOD: "Lincoln's nomination philosophy"

2. NEVER leave Subject or Object empty.
   If a verb has no clear object, skip that triplet.

3. For every active triplet [S, R, O], add its passive twin
   [O, R_passive, S].
   Use natural language for the passive relation.

4. Do not output duplicate facts.
   If two triplets express the same fact, keep the more specific one.

5. Relations must be grammatical verb phrases (prefer 2-6 words).

Example output:
[
  ["Jean Itard", "treated", "Wild Boy of Aveyron"],
  ["Wild Boy of Aveyron", "was treated by", "Jean Itard"],
  ["Wild Boy of Aveyron", "was caught in", "1798"],
  ["1798", "was the year of capture of", "Wild Boy of Aveyron"]
]
"""


INVALID_ENTITIES = {
    # placeholders
    "",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "-",
    # pronouns
    "he",
    "she",
    "it",
    "they",
    "we",
    "i",
    "you",
    "him",
    "her",
    "them",
    "us",
    "me",
    "his",
    "hers",
    "its",
    "their",
    "our",
    "my",
    "your",
    "himself",
    "herself",
    "itself",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    # vague filler words
    "most",
    "many",
    "some",
    "all",
    "both",
    "each",
    "other",
    "others",
    "another",
    "any",
    "few",
    "something",
    "someone",
    "anyone",
    "anything",
    "everything",
    "everyone",
    "nothing",
    "nobody",
    "one",
    "ones",
}


# ---------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------

def setup_openai_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY from the environment."""
    load_dotenv()
    return OpenAI()


# ---------------------------------------------------------------------
# Generic input loading
# ---------------------------------------------------------------------

def _get_nested_or_direct(record: dict[str, Any], key: str) -> Any:
    """Read a direct key. Kept separate so nested-key support can be added later."""
    return record.get(key)


def _detect_field(record: dict[str, Any], canonical_name: str) -> str | None:
    """Resolve a canonical field to an input field name.

    FIELD_MAP gets first priority. If that mapped field is absent in a
    particular record, the loader falls back to common field names so mixed
    schemas can still be processed safely.
    """
    explicit = FIELD_MAP.get(canonical_name)
    if explicit and explicit in record:
        return explicit

    for candidate in FIELD_CANDIDATES[canonical_name]:
        if candidate in record:
            return candidate
    return None


def _read_json_file(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Common wrapper structures.
        for key in ("data", "documents", "docs", "chunks", "records", "items"):
            if isinstance(data.get(key), list):
                return data[key]

        # Otherwise treat one JSON object as one record.
        return [data]

    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_jsonl_file(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_no}: {exc}"
                ) from exc
    return records


def _read_input_file(path: Path) -> list[Any]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _read_json_file(path)

    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_file(path)

    if suffix == ".txt":
        # First try JSONL because many existing CorpusHop datasets use .txt
        # while storing one JSON object per line.
        try:
            return _read_jsonl_file(path)
        except ValueError:
            # Fall back to treating the whole file as one plain-text document.
            text = path.read_text(encoding="utf-8-sig").strip()
            return [{"text": text, "source": path.stem}] if text else []

    raise ValueError(f"Unsupported file type: {path}")


def discover_input_files(input_path: str | Path, recursive: bool = False) -> list[Path]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported input extension {path.suffix}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [path]

    iterator = path.rglob("*") if recursive else path.glob("*")
    files = sorted(
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No supported input files found in {path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return files


def normalize_record(
    record: Any,
    *,
    fallback_id: str,
    source_file: Path,
) -> dict[str, Any] | None:
    """
    Convert an arbitrary input record to CorpusHop's canonical pre-extraction schema:
        id, topic, text
    """
    if isinstance(record, str):
        text = record.strip()
        if not text:
            return None
        return {
            "id": fallback_id,
            "topic": source_file.stem,
            "text": text,
        }

    if not isinstance(record, dict):
        return None

    text_field = _detect_field(record, "text")
    if not text_field:
        return None

    text = str(_get_nested_or_direct(record, text_field) or "").strip()
    if not text:
        return None

    id_field = _detect_field(record, "id")
    topic_field = _detect_field(record, "topic")

    raw_id = record.get(id_field) if id_field else None
    raw_topic = record.get(topic_field) if topic_field else None

    doc_id = raw_id if raw_id not in (None, "") else fallback_id
    topic = raw_topic if raw_topic not in (None, "") else source_file.stem

    return {
        "id": doc_id,
        "topic": topic,
        "text": text,
    }


def load_documents(
    input_path: str | Path,
    *,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    """
    Load and normalize all supported input files.

    The returned records always contain:
        id, topic, text
    """
    files = discover_input_files(input_path, recursive=recursive)

    docs: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    auto_counter = 0

    for file_path in files:
        raw_records = _read_input_file(file_path)

        for record in raw_records:
            while True:
                fallback_id = f"auto-{auto_counter:08d}"
                auto_counter += 1
                if fallback_id not in used_ids:
                    break

            doc = normalize_record(
                record,
                fallback_id=fallback_id,
                source_file=file_path,
            )
            if doc is None:
                continue

            # Make IDs unique across multiple input files.
            base_id = str(doc["id"])
            unique_id = base_id
            suffix = 1
            while unique_id in used_ids:
                unique_id = f"{base_id}__{suffix}"
                suffix += 1

            # Preserve the original type only when no collision occurred.
            doc["id"] = doc["id"] if unique_id == base_id else unique_id
            used_ids.add(str(doc["id"]))
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------
# Triple extraction and cleaning
# ---------------------------------------------------------------------

def _strip_code_fence(content: str) -> str:
    content = (content or "").strip()

    if content.startswith("```json"):
        content = content[len("```json") :]
    elif content.startswith("```"):
        content = content[len("```") :]

    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


def extract_triplets(
    text: str,
    client: OpenAI,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[list[str]]:
    """Extract raw [subject, relation, object] triplets from one text chunk."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )

    content = _strip_code_fence(response.choices[0].message.content or "")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed. Raw output:\n{content[:500]}")
        return []

    if not isinstance(parsed, list):
        print("  [WARN] Model output was valid JSON but not a list.")
        return []

    return parsed


def clean_triplets(
    raw: Iterable[Any],
    *,
    max_entity_words: int = 10,
) -> list[list[str]]:
    """
    Clean model output while preserving the original CorpusHop behavior:
    - lowercase
    - remove malformed/empty/vague entities
    - remove overly long entities
    - exact deduplication
    """
    seen: set[tuple[str, str, str]] = set()
    cleaned: list[list[str]] = []

    for triple in raw:
        if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
            continue

        subject = str(triple[0]).strip().lower()
        relation = str(triple[1]).strip().lower()
        obj = str(triple[2]).strip().lower()

        if not subject or not relation or not obj:
            continue

        if subject in INVALID_ENTITIES or obj in INVALID_ENTITIES:
            continue

        if len(subject.split()) > max_entity_words:
            continue
        if len(obj.split()) > max_entity_words:
            continue

        key = (subject, relation, obj)
        if key in seen:
            continue

        seen.add(key)
        cleaned.append([subject, relation, obj])

    return cleaned


def collect_entities(triplets: Iterable[list[str]]) -> list[str]:
    """Collect unique subject/object entities in canonical lowercase form."""
    return sorted(
        {
            entity.strip().lower()
            for triple in triplets
            for entity in (triple[0], triple[2])
            if entity.strip()
        }
    )


# ---------------------------------------------------------------------
# CorpusHop-compatible pipeline
# ---------------------------------------------------------------------

def load_processed_ids(output_file: Path) -> set[str]:
    """Read IDs already written to the output JSONL for safe resume."""
    processed: set[str] = set()

    if not output_file.exists():
        return processed

    with output_file.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] Skipping malformed existing output line "
                    f"{line_no} in {output_file}"
                )
                continue

            if "id" in record:
                processed.add(str(record["id"]))

    return processed


def extract_triplets_from_docs(
    docs: list[dict[str, Any]],
    client: OpenAI,
    *,
    output_file: str | Path = "docs_with_all_triples.jsonl",
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    resume: bool = True,
    max_entity_words: int = 10,
) -> Path:
    """
    Extract triples and write JSONL directly compatible with the generalized
    Bridge and Compositional CorpusHop pipelines.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(output_path) if resume else set()

    if not resume and output_path.exists():
        output_path.unlink()

    if processed_ids:
        print(f"Resume mode: {len(processed_ids)} records already processed.")

    mode = "a" if resume else "w"

    with output_path.open(mode, encoding="utf-8") as f:
        for index, doc in enumerate(docs, 1):
            doc_id = doc["id"]

            if str(doc_id) in processed_ids:
                continue

            text = str(doc.get("text", "")).strip()
            if not text:
                continue

            print(f"Processing {index}/{len(docs)} | id={doc_id}")

            raw_triplets = extract_triplets(
                text,
                client,
                model=model,
                temperature=temperature,
            )
            triplets = clean_triplets(
                raw_triplets,
                max_entity_words=max_entity_words,
            )
            entities = collect_entities(triplets)

            # Canonical schema consumed by the generalized CorpusHop notebooks.
            entry = {
                "id": doc_id,
                "topic": doc.get("topic"),
                "text": text,
                "extracts": triplets,
                "entities": entities,
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()

            processed_ids.add(str(doc_id))
            print(
                f"  -> {len(triplets)} triplets | "
                f"{len(entities)} entities saved"
            )

    print(f"\nDone. Output: {output_path}")
    return output_path


def run_pipeline(
    *,
    input_path: str | Path,
    output_file: str | Path,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    resume: bool = True,
    recursive: bool = False,
    max_entity_words: int = 10,
) -> Path:
    """End-to-end loader -> extraction -> canonical CorpusHop JSONL."""
    docs = load_documents(input_path, recursive=recursive)

    if not docs:
        raise ValueError("No usable documents were found in the input.")

    print(f"Loaded {len(docs)} usable documents.")
    print(f"Model: {model}")
    print(f"Output: {output_file}\n")

    client = setup_openai_client()

    return extract_triplets_from_docs(
        docs,
        client,
        output_file=output_file,
        model=model,
        temperature=temperature,
        resume=resume,
        max_entity_words=max_entity_words,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract knowledge-graph triplets and produce JSONL directly "
            "compatible with CorpusHop Bridge/Compositional pipelines."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory (.json/.jsonl/.ndjson/.txt).",
    )
    parser.add_argument(
        "--output",
        default="docs_with_all_triples.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature (default: 0).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Overwrite output instead of resuming from existing IDs.",
    )
    parser.add_argument(
        "--max-entity-words",
        type=int,
        default=10,
        help="Discard subject/object entities longer than this many words.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    run_pipeline(
        input_path=args.input,
        output_file=args.output,
        model=args.model,
        temperature=args.temperature,
        resume=not args.no_resume,
        recursive=args.recursive,
        max_entity_words=args.max_entity_words,
    )


if __name__ == "__main__":
    main()
