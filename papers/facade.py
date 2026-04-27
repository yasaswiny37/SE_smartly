"""
Facade Pattern — SmartConf Facades
====================================
SearchFacade   : single entry point for embedding + retrieval + LLM generation
IngestionFacade: single entry point for CSV / manual / BibTeX ingestion + indexing

Both facades hide all the complexity of ChromaDB, sentence-transformers, and the
local LLM from the Django views — views just call facade.search() or facade.ingest_*().
"""

import os
import csv
import json
import logging
import requests
from io import StringIO, BytesIO
from typing import List, Dict, Any, Optional

from django.conf import settings

from .strategies import RankedStrategy, BestMatchStrategy, SearchStrategy

logger = logging.getLogger(__name__)


# ─── Lazy singletons (avoid loading heavy models on every import) ──────────────

_embedding_model = None
_chroma_client   = None
_chroma_col      = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def _get_chroma_collection():
    global _chroma_client, _chroma_col
    if _chroma_col is None:
        import chromadb  # type: ignore
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        _chroma_col = _chroma_client.get_or_create_collection(
            name='papers',
            metadata={'hnsw:space': 'l2'},
        )
    return _chroma_col


# ─── SearchFacade ─────────────────────────────────────────────────────────────

class SearchFacade:
    """
    Facade that orchestrates:
      1. Embedding the user query
      2. Retrieving papers from ChromaDB using the chosen Strategy
      3. Building a prompt from retrieved abstracts
      4. Calling the local LLM for a grounded answer
      5. Filtering by conference / year if requested
    """

    STRATEGIES: Dict[str, SearchStrategy] = {
        'ranked'    : RankedStrategy(),
        'best_match': BestMatchStrategy(),
    }

    def search(
        self,
        query: str,
        mode: str = 'ranked',
        conference: str = '',
        year: str = '',
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Main search entry point called by the Django view.

        Returns:
          {
            'papers'  : [...],   # list of paper dicts with 'why' explanation
            'llm_answer': str,   # grounded LLM response
            'query'   : str,
            'mode'    : str,
          }
        """
        if not query or not query.strip():
            return {'papers': [], 'llm_answer': '', 'query': query, 'mode': mode, 'error': 'Query is required.'}

        try:
            model      = _get_embedding_model()
            collection = _get_chroma_collection()
        except Exception as e:
            logger.error("Failed to load embedding model or ChromaDB: %s", e)
            return {'papers': [], 'llm_answer': '', 'query': query, 'mode': mode, 'error': str(e)}

        # 1. Embed query
        query_embedding = model.encode(query).tolist()

        # 2. Build where filter for ChromaDB
        where = self._build_where(conference, year)

        # 3. Retrieve via Strategy
        strategy = self.STRATEGIES.get(mode, self.STRATEGIES['ranked'])
        try:
            if where:
                # ChromaDB requires passing where to query()
                raw = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances'],
                    where=where,
                )
                from .strategies import SearchStrategy
                papers = strategy._format_results(raw)
                if mode == 'ranked':
                    papers.sort(key=lambda p: p['score'], reverse=True)
                elif mode == 'best_match':
                    papers = papers[:1]
            else:
                papers = strategy.retrieve(query_embedding, collection, n_results)
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            papers = []

        # 4. Generate LLM answer from retrieved abstracts
        llm_answer = ''
        if papers:
            llm_answer = self._call_llm(query, papers)

        # 5. Add "why this matches" explanation per paper
        for i, paper in enumerate(papers):
            paper['rank']    = i + 1
            paper['is_best'] = (i == 0)
            paper['why']     = self._explain(query, paper)

        return {
            'papers'    : papers,
            'llm_answer': llm_answer,
            'query'     : query,
            'mode'      : mode,
        }

    # ── private helpers ──────────────────────────────────────────────────────

    def _build_where(self, conference: str, year: str) -> Optional[Dict]:
        """Build ChromaDB metadata filter."""
        conditions = []
        if conference:
            conditions.append({'conference': {'$eq': conference.upper()}})
        if year:
            conditions.append({'year': {'$eq': year}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {'$and': conditions}

    def _call_llm(self, query: str, papers: List[Dict]) -> str:
        """POST to the local LLM and return its response text."""
        abstracts = '\n\n'.join(
            f"[{i+1}] {p['title']} ({p['conference']} {p['year']})\n{p['abstract']}"
            for i, p in enumerate(papers[:3])  # top-3 only
        )
        prompt = (
            f"You are a research assistant. Answer the following question using ONLY "
            f"the provided paper abstracts. Do not invent any information.\n\n"
            f"Question: {query}\n\n"
            f"Abstracts:\n{abstracts}\n\n"
            f"Answer:"
        )
        try:
            resp = requests.post(
                settings.LLM_ENDPOINT,
                json={'prompt': prompt, 'max_tokens': settings.LLM_MAX_TOKENS},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get('response', '')
        except Exception as e:
            logger.warning("LLM request failed: %s", e)
        return ''

    def _explain(self, query: str, paper: Dict) -> str:
        """Generate a brief 'why this matches' explanation from the abstract."""
        # Simple keyword overlap approach (no extra LLM call needed)
        query_words = set(query.lower().split())
        abstract    = paper['abstract'].lower()
        matches     = [w for w in query_words if len(w) > 3 and w in abstract]
        if matches:
            highlighted = ', '.join(f'"{w}"' for w in matches[:4])
            return (
                f"Matches your query on {highlighted}. "
                f"Semantic similarity score: {paper['score']:.2f}."
            )
        return f"Semantic similarity score: {paper['score']:.2f}."


# ─── IngestionFacade ──────────────────────────────────────────────────────────

class IngestionFacade:
    """
    Facade that orchestrates:
      1. Parsing CSV / JSON / BibTeX / manual form data
      2. Validating required fields (title, authors, abstract, conference, year)
      3. Saving to SQLite via Django ORM
      4. Embedding abstract and storing in ChromaDB
    """

    REQUIRED = ['title', 'authors', 'abstract', 'conference', 'year']

    # ── CSV ingestion ─────────────────────────────────────────────────────────

    def ingest_csv(self, file_obj) -> Dict[str, Any]:
        """
        Parse and ingest a CSV file.
        Expected columns: title, authors, abstract, conference, year, doi_url (optional)
        Returns: {'saved': int, 'skipped': int, 'errors': [str]}
        """
        saved, skipped, errors = 0, 0, []
        try:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            reader = csv.DictReader(StringIO(content))
            # Normalise column names
            rows = []
            for row in reader:
                normalised = {k.strip().lower(): v.strip() for k, v in row.items()}
                rows.append(normalised)
        except Exception as e:
            return {'saved': 0, 'skipped': 0, 'errors': [f"CSV parse error: {e}"]}

        for i, row in enumerate(rows, start=2):  # row 1 = header
            err = self._validate(row)
            if err:
                errors.append(f"Row {i}: {err}")
                skipped += 1
                continue
            ok, msg = self._save_paper(row)
            if ok:
                saved += 1
            else:
                errors.append(f"Row {i}: {msg}")
                skipped += 1

        return {'saved': saved, 'skipped': skipped, 'errors': errors}

    # ── JSON ingestion ────────────────────────────────────────────────────────

    def ingest_json(self, file_obj) -> Dict[str, Any]:
        saved, skipped, errors = 0, 0, []
        try:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
        except Exception as e:
            return {'saved': 0, 'skipped': 0, 'errors': [f"JSON parse error: {e}"]}

        for i, row in enumerate(data, start=1):
            normalised = {k.strip().lower(): str(v).strip() for k, v in row.items()}
            err = self._validate(normalised)
            if err:
                errors.append(f"Record {i}: {err}")
                skipped += 1
                continue
            ok, msg = self._save_paper(normalised)
            if ok:
                saved += 1
            else:
                errors.append(f"Record {i}: {msg}")
                skipped += 1

        return {'saved': saved, 'skipped': skipped, 'errors': errors}

    # ── BibTeX ingestion ──────────────────────────────────────────────────────

    def ingest_bibtex(self, file_obj) -> Dict[str, Any]:
        """
        Simple BibTeX parser — extracts title, author, abstract, booktitle/journal, year.
        """
        saved, skipped, errors = 0, 0, []
        try:
            content = file_obj.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            entries = self._parse_bibtex(content)
        except Exception as e:
            return {'saved': 0, 'skipped': 0, 'errors': [f"BibTeX parse error: {e}"]}

        for i, entry in enumerate(entries, start=1):
            row = {
                'title'      : entry.get('title', '').strip('{}'),
                'authors'    : entry.get('author', '').replace('\n', ' '),
                'abstract'   : entry.get('abstract', ''),
                'conference' : entry.get('booktitle', entry.get('journal', '')).strip('{}'),
                'year'       : entry.get('year', ''),
                'doi_url'    : entry.get('url', entry.get('doi', '')),
            }
            err = self._validate(row)
            if err:
                errors.append(f"Entry {i}: {err}")
                skipped += 1
                continue
            ok, msg = self._save_paper(row)
            if ok:
                saved += 1
            else:
                errors.append(f"Entry {i}: {msg}")
                skipped += 1

        return {'saved': saved, 'skipped': skipped, 'errors': errors}

    # ── Manual entry ──────────────────────────────────────────────────────────

    def ingest_manual(self, data: Dict) -> Dict[str, Any]:
        """Ingest a single paper from the manual entry form."""
        normalised = {k.strip().lower(): str(v).strip() for k, v in data.items()}
        err = self._validate(normalised)
        if err:
            return {'saved': 0, 'skipped': 1, 'errors': [err]}
        ok, msg = self._save_paper(normalised)
        if ok:
            return {'saved': 1, 'skipped': 0, 'errors': []}
        return {'saved': 0, 'skipped': 1, 'errors': [msg]}

    # ── shared helpers ────────────────────────────────────────────────────────

    def _validate(self, row: Dict) -> str:
        """Return an error message string, or '' if valid."""
        for field in self.REQUIRED:
            if not row.get(field, '').strip():
                return f"Missing required field: '{field}'"
        try:
            year = int(row['year'])
            if not (1900 <= year <= 2100):
                return f"Year out of range: {year}"
        except ValueError:
            return f"Invalid year: {row['year']}"
        return ''

    def _save_paper(self, row: Dict):
        """Save to SQLite and index in ChromaDB. Returns (success: bool, message: str)."""
        from .models import Paper, Conference
        try:
            conf_name = row['conference'].strip().upper()
            conference, _ = Conference.objects.get_or_create(
                name=conf_name,
                defaults={'full_name': conf_name},
            )
            # Avoid exact duplicates
            if Paper.objects.filter(
                title__iexact=row['title'],
                conference=conference,
                year=int(row['year']),
            ).exists():
                return False, f"Duplicate: '{row['title']}' already exists."

            paper = Paper.objects.create(
                title      = row['title'],
                authors    = row['authors'],
                abstract   = row['abstract'],
                conference = conference,
                year       = int(row['year']),
                doi_url    = row.get('doi_url', ''),
            )
            # Index in ChromaDB
            self._index_paper(paper)
            return True, ''
        except Exception as e:
            logger.error("Error saving paper: %s", e)
            return False, str(e)

    def _index_paper(self, paper):
        """Embed abstract and upsert into ChromaDB."""
        try:
            model      = _get_embedding_model()
            collection = _get_chroma_collection()
            embedding  = model.encode(paper.abstract).tolist()
            collection.upsert(
                ids        = [str(paper.id)],
                embeddings = [embedding],
                documents  = [paper.abstract],
                metadatas  = [{
                    'title'      : paper.title,
                    'authors'    : paper.authors,
                    'conference' : paper.conference.name,
                    'year'       : str(paper.year),
                    'doi_url'    : paper.doi_url,
                }],
            )
            paper.indexed = True
            paper.save(update_fields=['indexed'])
        except Exception as e:
            logger.warning("ChromaDB indexing failed for paper %s: %s", paper.id, e)

    def _parse_bibtex(self, content: str) -> List[Dict]:
        """Minimal BibTeX parser — no external library needed."""
        import re
        entries = []
        # Match each @type{key, ...} block
        pattern = re.compile(r'@\w+\s*\{[^@]*\}', re.DOTALL)
        for match in pattern.findall(content):
            entry = {}
            # Extract field = {value} or field = "value"
            fields = re.findall(r'(\w+)\s*=\s*[\{"](.*?)[\}"](?:\s*,|\s*\})', match, re.DOTALL)
            for key, val in fields:
                entry[key.lower()] = val.strip()
            if entry:
                entries.append(entry)
        return entries

    def reindex_all(self) -> int:
        """Re-embed and re-index all papers (admin utility)."""
        from .models import Paper
        count = 0
        for paper in Paper.objects.all():
            self._index_paper(paper)
            count += 1
        return count