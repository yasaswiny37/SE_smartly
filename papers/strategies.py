"""
Strategy Pattern — Search Retrieval Strategies
================================================
SearchStrategy   : abstract base (interface)
RankedStrategy   : returns top-k results ranked by similarity score
BestMatchStrategy: returns only the single best result
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SearchStrategy(ABC):
    """Abstract base strategy — defines the retrieval interface."""

    @abstractmethod
    def retrieve(self, query_embedding, chroma_collection, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve papers from ChromaDB.

        Args:
            query_embedding : list of floats from the embedding model
            chroma_collection: ChromaDB collection object
            n_results        : max number of results to fetch

        Returns:
            List of dicts with keys: id, title, authors, abstract, conference, year, doi_url, score
        """
        pass

    def _format_results(self, chroma_results) -> List[Dict[str, Any]]:
        """Convert raw ChromaDB query results into a clean list of dicts."""
        papers = []
        if not chroma_results or not chroma_results['ids']:
            return papers

        ids       = chroma_results['ids'][0]
        documents = chroma_results['documents'][0]
        metadatas = chroma_results['metadatas'][0]
        distances = chroma_results['distances'][0]

        for i, doc_id in enumerate(ids):
            meta = metadatas[i]
            # ChromaDB distance is L2; convert to similarity score 0-1
            score = round(1 / (1 + distances[i]), 4)
            papers.append({
                'id'         : doc_id,
                'title'      : meta.get('title', ''),
                'authors'    : meta.get('authors', ''),
                'abstract'   : documents[i],
                'conference' : meta.get('conference', ''),
                'year'       : meta.get('year', ''),
                'doi_url'    : meta.get('doi_url', ''),
                'score'      : score,
            })
        return papers


class RankedStrategy(SearchStrategy):
    """
    Returns the top-k papers ranked by semantic similarity (highest score first).
    Used for the 'Ranked' view on the search page.
    """

    def retrieve(self, query_embedding, chroma_collection, n_results: int = 5) -> List[Dict[str, Any]]:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
        )
        papers = self._format_results(results)
        # Sort descending by score (highest similarity first)
        papers.sort(key=lambda p: p['score'], reverse=True)
        return papers


class BestMatchStrategy(SearchStrategy):
    """
    Returns only the single best-matching paper.
    Used for the 'Best Match' view on the search page.
    """

    def retrieve(self, query_embedding, chroma_collection, n_results: int = 5) -> List[Dict[str, Any]]:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=1,          # only need 1
            include=['documents', 'metadatas', 'distances'],
        )
        papers = self._format_results(results)
        return papers[:1]         # guaranteed single result