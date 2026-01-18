#!/usr/bin/env python3
"""
legalassist_civix_test.py

A minimal end-to-end test harness for your hackathon:
- Pulls official docs from:
  - BC Laws CiviX (XML) for BC legislation (rental/work/driving-law)
  - PDFs (ICBC + Justice Laws) for driving manuals + immigration
- Chunks + embeds them (local LLM embeddings, e.g., Ollama)
- Stores chunks + vectors in SQLite
- Answers questions using ONLY the stored doc chunks and prints Sources

Prereqs:
  pip install requests pypdf

Local LLM (recommended: Ollama):
  - chat model: mistral:7b (or any)
  - embed model: nomic-embed-text (or bge-m3)

Env vars (optional):
  LLM_BASE_URL=http://localhost:11434
  LLM_CHAT_MODEL=mistral:7b
  LLM_EMBED_MODEL=nomic-embed-text
  DB_PATH=legalassist_test.db

Usage:
  # Ingest one category
  python legalassist_civix_test.py ingest rental
  python legalassist_civix_test.py ingest driving
  python legalassist_civix_test.py ingest workbc
  python legalassist_civix_test.py ingest immigration

  # Ask a question (answers ONLY from ingested docs)
  python legalassist_civix_test.py ask rental "How much notice must a landlord give before entering?"
  python legalassist_civix_test.py ask driving "At a four-way stop, who goes first if two vehicles arrive together?"
  python legalassist_civix_test.py ask workbc "When is overtime pay required in BC?"
  python legalassist_civix_test.py ask immigration "What are the objectives of the IRPA?"

Notes:
- This is a hackathon-grade baseline. It intentionally avoids “web fallback” and “auto-updates”.
- If you want, you can later add Tavily fallback when retrieval is weak.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from pypdf import PdfReader


# ----------------------------
# Config
# ----------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "mistral:latest")
LLM_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "nomic-embed-text:latest")
DB_PATH = os.getenv("DB_PATH", "legalassist_test.db")

TOP_K = 6
MIN_SIMILARITY = 0.18  # raise to be stricter, lower to be more permissive


# Official sources (hackathon starter set)
SOURCES: Dict[str, List[dict]] = {
    "rental": [
        {
            "title": "Residential Tenancy Act (BC)",
            "kind": "civix_xml",
            "source_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/02078_01",
            "fetch_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/02078_01/xml",
        },
        {
            "title": "Residential Tenancy Regulation (BC Reg. 477/2003)",
            "kind": "civix_xml",
            "source_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/10_477_2003",
            "fetch_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/10_477_2003/xml",
        },
    ],
    "workbc": [
        {
            "title": "Employment Standards Act (BC)",
            "kind": "civix_xml",
            "source_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/00_96113_01",
            "fetch_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/00_96113_01/xml",
        },
        {
            "title": "Employment Standards Regulation (B.C. Reg. 396/95)",
            "kind": "civix_xml",
            "source_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/396_95",
            "fetch_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/396_95/xml",
        },
    ],
    "driving": [
        {
            "title": "Motor Vehicle Act (BC)",
            "kind": "civix_xml",
            "source_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/96318_01",
            "fetch_url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/96318_01/xml",
        },
        {
            "title": "ICBC: Learn to Drive Smart (Driver’s Manual)",
            "kind": "pdf",
            # This is a direct PDF used on the ICBC site (URL can change over time).
            "source_url": "https://downloads.ctfassets.net/nnc41duedoho/63cHBOAVpOAQGOOMBFhFbL/0120c57c3c706956bd3e410e179642bd/driver-full.pdf",
            "fetch_url": "https://downloads.ctfassets.net/nnc41duedoho/63cHBOAVpOAQGOOMBFhFbL/0120c57c3c706956bd3e410e179642bd/driver-full.pdf",
        },
    ],
    "immigration": [
        {
            "title": "Immigration and Refugee Protection Act (IRPA) (Canada)",
            "kind": "pdf",
            "source_url": "https://laws.justice.gc.ca/PDF/I-2.5.pdf",
            "fetch_url": "https://laws.justice.gc.ca/PDF/I-2.5.pdf",
        },
        {
            "title": "Immigration and Refugee Protection Regulations (IRPR) (Canada)",
            "kind": "pdf",
            "source_url": "https://laws-lois.justice.gc.ca/pdf/sor-2002-227.pdf",
            "fetch_url": "https://laws-lois.justice.gc.ca/pdf/sor-2002-227.pdf",
        },
    ],
}


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Chunk:
    id: int
    category: str
    title: str
    source_url: str
    locator: str  # e.g., "s. 29" or "pp. 12–13"
    text: str
    embedding: List[float]


# ----------------------------
# Local LLM client (Ollama-style HTTP)
# ----------------------------

def _post_json(url: str, payload: dict, timeout_s: int = 180) -> dict:
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def llm_embed_one(text: str) -> List[float]:
    """
    Uses local embeddings endpoint (Ollama: /api/embeddings).
    """
    data = _post_json(
        f"{LLM_BASE_URL}/api/embeddings",
        {"model": LLM_EMBED_MODEL, "prompt": text},
        timeout_s=180,
    )
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("Embeddings endpoint returned no embedding. Check LLM_EMBED_MODEL.")
    return [float(x) for x in emb]


def llm_chat(messages: List[dict], temperature: float = 0.2) -> str:
    # Convert messages into one prompt for /api/generate
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        prompt_parts.append(f"{role}:\n{content}")
    prompt = "\n\n".join(prompt_parts) + "\n\nASSISTANT:\n"

    # Retry once if something transient happens
    for attempt in (1, 2):
        try:
            data = _post_json(
                f"{LLM_BASE_URL}/api/generate",
                {
                    "model": LLM_CHAT_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout_s=240,
            )
            out = data.get("response")
            if not isinstance(out, str):
                raise RuntimeError(f"Unexpected generate response: {data}")
            return out.strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.3)

# ----------------------------
# SQLite storage
# ----------------------------

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            source_url TEXT NOT NULL,
            locator TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);
        """
    )
    conn.commit()


def db_clear_category(conn: sqlite3.Connection, category: str) -> None:
    conn.execute("DELETE FROM chunks WHERE category = ?;", (category,))
    conn.commit()


def db_insert_chunk(conn: sqlite3.Connection, category: str, title: str, source_url: str, locator: str, text: str, embedding: List[float]) -> None:
    conn.execute(
        """
        INSERT INTO chunks(category, title, source_url, locator, text, embedding_json)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (category, title, source_url, locator, text, json.dumps(embedding)),
    )


def db_load_category(conn: sqlite3.Connection, category: str) -> List[Chunk]:
    rows = conn.execute(
        "SELECT id, category, title, source_url, locator, text, embedding_json FROM chunks WHERE category = ?;",
        (category,),
    ).fetchall()
    chunks: List[Chunk] = []
    for (id_, cat, title, url, loc, text, emb_json) in rows:
        chunks.append(Chunk(
            id=int(id_),
            category=str(cat),
            title=str(title),
            source_url=str(url),
            locator=str(loc),
            text=str(text),
            embedding=[float(x) for x in json.loads(emb_json)],
        ))
    return chunks


# ----------------------------
# Utility: similarity + chunking
# ----------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[str]:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return [c for c in chunks if c.strip()]


# ----------------------------
# Fetch + parse sources
# ----------------------------

def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=180, headers={"User-Agent": "legalassist-hackathon/0.1"})
    r.raise_for_status()
    return r.content


# --- PDF parsing ---

def pdf_to_page_texts(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io_bytes(pdf_bytes))
    out: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        out.append(normalize_ws(t))
    return out


def io_bytes(b: bytes):
    import io
    return io.BytesIO(b)


def ingest_pdf(category: str, title: str, source_url: str, fetch_url: str, conn: sqlite3.Connection) -> int:
    pdf_bytes = fetch_bytes(fetch_url)
    reader = PdfReader(io_bytes(pdf_bytes))

    n_chunks = 0
    for idx, page in enumerate(reader.pages, start=1):
        t = normalize_ws(page.extract_text() or "")
        if not t:
            continue
        # Chunk within page if needed
        subchunks = chunk_text(t, max_chars=1600, overlap=200)
        for sc in subchunks:
            locator = f"p. {idx}"
            emb = llm_embed_one(sc[:4000])  # keep embedding input bounded
            db_insert_chunk(conn, category, title, source_url, locator, sc, emb)
            n_chunks += 1
    conn.commit()
    return n_chunks


# --- CiviX XML parsing ---
# CiviX XML uses namespaces like:
#   bcl:section, bcl:num, bcl:marginalnote, bcl:text
BCL_NS = "http://www.gov.bc.ca/2013/bclegislation"
ACT_NS = "http://www.gov.bc.ca/2013/legislation/act"
REG_NS = "http://www.gov.bc.ca/2013/legislation/regulation"
NSMAP = {"bcl": BCL_NS, "act": ACT_NS, "reg": REG_NS}

def _findall(elem: ET.Element, path: str) -> List[ET.Element]:
    return elem.findall(path, namespaces=NSMAP)

def _findtext(elem: ET.Element, path: str) -> str:
    node = elem.find(path, namespaces=NSMAP)
    if node is None:
        return ""
    return "".join(node.itertext()).strip()

def civix_xml_to_sections(xml_bytes: bytes) -> List[Tuple[str, str]]:
    """
    Returns [(locator, section_text), ...]
    Locator tries to be: "s. <num>" or similar.
    """
    root = ET.fromstring(xml_bytes)

    sections = []
    for sec in root.iterfind(".//bcl:section", namespaces=NSMAP):
        num = _findtext(sec, "./bcl:num")
        note = _findtext(sec, "./bcl:marginalnote")
        # Collect all text within this section (excluding nested section numbers if any)
        body = normalize_ws(" ".join(sec.itertext()))
        # Body often repeats num/note; keep it but make it readable
        header = ""
        if num:
            header += f"Section {num}"
        if note:
            header += f" — {note}" if header else note
        # Keep a cleaner combined text:
        combined = normalize_ws((header + "\n" + body).strip())
        if not combined:
            continue
        locator = f"s. {num}" if num else "section"
        sections.append((locator, combined))

    # If we fail to find structured sections, fallback to whole-doc chunking
    if not sections:
        plain = normalize_ws(" ".join(root.itertext()))
        for i, c in enumerate(chunk_text(plain, max_chars=1800, overlap=250), start=1):
            sections.append((f"chunk {i}", c))

    return sections


def ingest_civix_xml(category: str, title: str, source_url: str, fetch_url: str, conn: sqlite3.Connection) -> int:
    xml_bytes = fetch_bytes(fetch_url)
    sections = civix_xml_to_sections(xml_bytes)

    n_chunks = 0
    for locator, text in sections:
        # If a section is huge, split it but keep same locator
        for sc in chunk_text(text, max_chars=1800, overlap=250):
            emb = llm_embed_one(sc[:4000])
            db_insert_chunk(conn, category, title, source_url, locator, sc, emb)
            n_chunks += 1

    conn.commit()
    return n_chunks


# ----------------------------
# Ingest / Ask
# ----------------------------

def ingest_category(category: str) -> None:
    if category not in SOURCES:
        raise SystemExit(f"Unknown category: {category}. Choose one of: {', '.join(SOURCES)}")

    conn = db_connect()
    db_init(conn)

    print(f"[ingest] Clearing existing chunks for category: {category}")
    db_clear_category(conn, category)

    total = 0
    for src in SOURCES[category]:
        print(f"[ingest] Fetching: {src['title']} ({src['kind']})")
        t0 = time.time()
        if src["kind"] == "civix_xml":
            n = ingest_civix_xml(category, src["title"], src["source_url"], src["fetch_url"], conn)
        elif src["kind"] == "pdf":
            n = ingest_pdf(category, src["title"], src["source_url"], src["fetch_url"], conn)
        else:
            raise RuntimeError(f"Unsupported source kind: {src['kind']}")
        dt = time.time() - t0
        total += n
        print(f"[ingest]  -> stored {n} chunks in {dt:.1f}s")

    print(f"[ingest] Done. Stored total chunks for {category}: {total}")
    conn.close()


def answer_from_docs(category: str, question: str) -> None:
    conn = db_connect()
    db_init(conn)

    chunks = db_load_category(conn, category)
    if not chunks:
        print(f"No chunks found for category '{category}'. Run: ingest {category}")
        return

    q_emb = llm_embed_one(question)
    scored: List[Tuple[float, Chunk]] = []
    for c in chunks:
        sim = cosine_similarity(q_emb, c.embedding)
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:TOP_K]
    if not top or top[0][0] < MIN_SIMILARITY:
        print("Not found in the currently ingested documents for this category.")
        return

    # Build deduped source labels
    sources: List[Tuple[str, str, str]] = []  # (title, locator, url)
    seen = set()
    for sim, c in top:
        key = (c.title, c.locator, c.source_url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(key)

    # Provide context to the model
    source_lines = []
    for i, (title, locator, url) in enumerate(sources, start=1):
        source_lines.append(f"[S{i}] {title} ({locator})")

    excerpt_blocks = []
    for sim, c in top:
        sid = 1
        for i, (title, locator, url) in enumerate(sources, start=1):
            if title == c.title and locator == c.locator and url == c.source_url:
                sid = i
                break
        excerpt_blocks.append(f"---\nSource [S{sid}] ({c.title}, {c.locator})\n{c.text}")

    system = (
        "You are LegalAssist AI.\n"
        "Answer ONLY using the provided excerpts.\n"
        "When you state a rule or fact, cite it using [S#].\n"
        "If the excerpts do not contain enough info to answer, output exactly: NOT_FOUND.\n"
        "Do not invent sources, links, or legal rules."
    )
    user = (
        f"Category: {category}\n"
        f"Question: {question}\n\n"
        f"Available Sources:\n" + "\n".join(source_lines) + "\n\n"
        f"Excerpts:\n" + "\n".join(excerpt_blocks)
    )

    t0 = time.time()
    draft = llm_chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    dt = time.time() - t0

    if draft.strip() == "NOT_FOUND" or "NOT_FOUND" in draft:
        print("Not found in the currently ingested documents for this category.")
        return

    print(draft)
    print("\nSources:")
    for i, (title, locator, url) in enumerate(sources, start=1):
        # For BC Laws docs, you can often add a section anchor in future (optional).
        print(f"- [S{i}] {title} ({locator}) — {url}")

    print(f"\n(took {dt:.1f}s, top similarity={top[0][0]:.3f})")
    conn.close()


# ----------------------------
# CLI
# ----------------------------

def main(argv: List[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(
            "Usage:\n"
            "  python legalassist_civix_test.py ingest <category>\n"
            "  python legalassist_civix_test.py ask <category> <question>\n\n"
            f"Categories: {', '.join(SOURCES.keys())}\n\n"
            "Example:\n"
            "  python legalassist_civix_test.py ingest rental\n"
            "  python legalassist_civix_test.py ask rental \"How much notice must a landlord give before entering?\"\n"
        )
        return 0

    cmd = argv[1].lower()
    if cmd == "ingest":
        if len(argv) != 3:
            print("Usage: ingest <category>")
            return 2
        ingest_category(argv[2].lower())
        return 0

    if cmd == "ask":
        if len(argv) < 4:
            print("Usage: ask <category> <question>")
            return 2
        category = argv[2].lower()
        question = " ".join(argv[3:])
        answer_from_docs(category, question)
        return 0

    print(f"Unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
