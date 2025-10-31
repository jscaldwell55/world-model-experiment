"""
Simple embedding utility for ACE bullet retrieval.

Uses sentence-transformers for local embeddings or falls back to
simple TF-IDF if not available.
"""

import numpy as np
from typing import List, Tuple, Optional
import json
import os


class EmbeddingModel:
    """
    Embedding model for semantic similarity.

    Tries to use sentence-transformers if available, otherwise falls back
    to simple TF-IDF based similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: Name of sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.use_transformer = False

        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_transformer = True
            print(f"Loaded sentence-transformer model: {model_name}")
        except ImportError:
            print("sentence-transformers not available, using TF-IDF fallback")
            self._init_tfidf()

    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer as fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=384)
            print("Using TF-IDF for embeddings")
        except ImportError:
            print("Warning: sklearn not available, embeddings disabled")
            self.vectorizer = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        if self.use_transformer:
            return self.model.encode(texts, convert_to_numpy=True)
        elif self.vectorizer is not None:
            # For TF-IDF, we need consistent vocabulary
            # Store the vocabulary if this is the first call
            if not hasattr(self, '_fitted'):
                self.vectorizer.fit(texts)
                self._fitted = True

            # Transform using the fitted vocabulary
            embeddings = self.vectorizer.transform(texts).toarray()
            return embeddings
        else:
            # No embedding available, return random vectors
            return np.random.randn(len(texts), 384)

    def similarity(self, query_embedding: np.ndarray,
                   doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding, shape (embedding_dim,)
            doc_embeddings: Document embeddings, shape (n_docs, embedding_dim)

        Returns:
            Similarity scores, shape (n_docs,)
        """
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities


class BulletRetriever:
    """
    Retrieves most relevant bullets from playbook using embeddings.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None,
                 top_k: int = 5,
                 utility_weight: float = 0.3,
                 recency_weight: float = 0.1,
                 recency_decay: float = 0.95):
        """
        Initialize retriever.

        Args:
            embedding_model: Embedding model to use (creates default if None)
            top_k: Number of bullets to retrieve per section
            utility_weight: Weight for helpful-harmful utility in scoring (0-1)
            recency_weight: Weight for recency in scoring (0-1)
            recency_decay: Decay factor for recency (0-1, lower = faster decay)
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k
        self.utility_weight = utility_weight
        self.recency_weight = recency_weight
        self.recency_decay = recency_decay
        self.bullet_cache = {}  # Cache bullet embeddings

    def retrieve_bullets(self, query: str, playbook: dict,
                        top_k: Optional[int] = None,
                        current_step: Optional[int] = None) -> dict:
        """
        Retrieve most relevant bullets from each section.

        Uses combined scoring:
        - Semantic similarity to query
        - Utility (helpful_count - harmful_count)
        - Recency (decay based on last_used_step)

        Args:
            query: Query text (observation + recent history)
            playbook: Full playbook dictionary
            top_k: Override default top_k if specified
            current_step: Current step number (for recency calculation)

        Returns:
            Filtered playbook with only top-k bullets per section
        """
        k = top_k or self.top_k

        # Filter each section
        filtered_playbook = {}
        for section_name, bullets in playbook.items():
            if not bullets:
                filtered_playbook[section_name] = []
                continue

            # Get bullet texts
            bullet_texts = [b['content'] for b in bullets]

            # Embed query and bullets together for consistent vocabulary
            all_texts = [query] + bullet_texts
            all_embeddings = self.embedding_model.embed(all_texts)

            query_embedding = all_embeddings[0]
            bullet_embeddings = all_embeddings[1:]

            # Compute semantic similarities (0-1 range after normalization)
            similarities = self.embedding_model.similarity(
                query_embedding,
                bullet_embeddings
            )
            # Normalize to 0-1 range
            if similarities.max() > similarities.min():
                similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

            # Compute combined scores
            combined_scores = []
            for i, bullet in enumerate(bullets):
                # Semantic similarity component
                sim_score = similarities[i]

                # Utility component (helpful - harmful, normalized)
                helpful = bullet.get('helpful_count', 0)
                harmful = bullet.get('harmful_count', 0)
                utility = helpful - harmful
                # Normalize utility to 0-1 range (assuming max utility ~10)
                utility_score = max(0, min(1, (utility + 5) / 10))  # Shift and scale

                # Recency component
                recency_score = 0.5  # Default neutral score
                if current_step is not None and bullet.get('last_used_step') is not None:
                    steps_ago = current_step - bullet['last_used_step']
                    recency_score = self.recency_decay ** steps_ago

                # Combine scores with weights
                semantic_weight = 1.0 - self.utility_weight - self.recency_weight
                combined = (
                    semantic_weight * sim_score +
                    self.utility_weight * utility_score +
                    self.recency_weight * recency_score
                )
                combined_scores.append(combined)

            # Get top-k indices by combined score
            top_indices = np.argsort(combined_scores)[::-1][:k]

            # Filter bullets
            filtered_playbook[section_name] = [
                bullets[i] for i in top_indices
            ]

        return filtered_playbook

    def _get_or_compute_embeddings(self, texts: List[str],
                                   ids: List[str]) -> np.ndarray:
        """
        Get cached embeddings or compute new ones.

        Args:
            texts: List of text strings
            ids: List of bullet IDs

        Returns:
            Array of embeddings
        """
        embeddings = []
        to_embed = []
        to_embed_indices = []

        for i, (text, bullet_id) in enumerate(zip(texts, ids)):
            if bullet_id in self.bullet_cache:
                embeddings.append(self.bullet_cache[bullet_id])
            else:
                to_embed.append(text)
                to_embed_indices.append(i)

        # Compute missing embeddings
        if to_embed:
            new_embeddings = self.embedding_model.embed(to_embed)
            for i, bullet_id in enumerate([ids[j] for j in to_embed_indices]):
                self.bullet_cache[bullet_id] = new_embeddings[i]
                embeddings.insert(to_embed_indices[i], new_embeddings[i])

        return np.array(embeddings)

    def clear_cache(self):
        """Clear embedding cache."""
        self.bullet_cache = {}
