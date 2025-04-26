"""
name_match_manager.py

Manages synonym-based name matching for table identification in the TableIdentifier-v2.1 application.
Loads synonyms from a JSON file, matches query terms to tables using SpaCy similarity, and updates
synonyms based on user feedback. Caches matches in Redis for efficient retrieval.

Dependencies:
- spacy: For similarity scoring (en_core_web_md model).
- json, os, logging: For file handling and logging.
- redis: For caching matches.
"""

import json
import os
import spacy
from typing import Dict, List
import logging
import redis

class NameMatchManager:
    """
    Manages synonym-based name matching for table identification.

    Loads synonyms from a JSON file, matches queries to tables using SpaCy similarity,
    and updates synonyms based on feedback. Maintains a matches dictionary for term-to-table
    mappings and caches it in Redis.

    Attributes:
        logger (logging.Logger): Logger for name matching operations.
        db_name (str): Database name (e.g., BikeStores).
        redis_client (redis.Redis): Redis client for caching.
        synonyms_file (str): Path to synonyms JSON file.
        synonyms (Dict[str, List[str]]): Mapping of tables to synonym terms.
        matches (Dict[str, Dict[str, float]]): Mapping of terms to tables and similarity scores.
        nlp (spacy.language.Language): SpaCy NLP model (en_core_web_md).
    """

    def __init__(self, db_name: str):
        """
        Initialize NameMatchManager with database name.

        Args:
            db_name (str): Database name.
        """
        self.logger = logging.getLogger("name_match_manager")
        self.db_name = db_name
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.synonyms_file = f"app-config/{db_name}/BikeStores_synonyms.json"
        os.makedirs(f"app-config/{db_name}", exist_ok=True)
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self.synonyms = self._load_synonyms()
        self.matches = self._load_matches()
        self.logger.debug(f"Initialized NameMatchManager for {db_name}")

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """
        Load synonyms from JSON file or initialize empty dictionary.

        Returns:
            Dict[str, List[str]]: Mapping of tables to synonym terms.
        """
        try:
            if os.path.exists(self.synonyms_file):
                with open(self.synonyms_file, 'r') as f:
                    synonyms = json.load(f)
                    valid_synonyms = {k: v for k, v in synonyms.items() if isinstance(v, list)}
                    self.logger.debug(f"Loaded {len(valid_synonyms)} valid synonyms from {self.synonyms_file}")
                    return valid_synonyms
            self.logger.debug(f"No synonyms file found at {self.synonyms_file}, initializing empty")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading synonyms: {e}")
            return {}

    def _load_matches(self) -> Dict[str, Dict[str, float]]:
        """
        Load matches from Redis or initialize empty dictionary.

        Returns:
            Dict[str, Dict[str, float]]: Mapping of terms to tables and similarity scores.
        """
        try:
            matches_data = self.redis_client.get(f"{self.db_name}:name_matches")
            if matches_data:
                matches = json.loads(matches_data)
                self.logger.debug(f"Loaded matches from Redis for {self.db_name}")
                return matches
            self.logger.debug(f"No matches in Redis for {self.db_name}, initializing empty")
            return {}
        except redis.RedisError as e:
            self.logger.error(f"Redis error loading matches: {e}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for matches: {e}")
            return {}

    def get_matches(self, query: str) -> Dict[str, float]:
        """
        Match query terms to tables using synonym similarity.

        Args:
            query (str): Preprocessed query text.

        Returns:
            Dict[str, float]: Dictionary of table names and similarity scores.
        """
        self.logger.debug(f"Matching query: {query}")
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, skipping name matching")
            return {}

        try:
            query_lower = query.lower()
            query_doc = self.nlp(query_lower)
            query_terms = [token.lemma_ for token in query_doc if not token.is_stop and token.is_alpha]
            matches = {}

            for table, synonyms in self.synonyms.items():
                for synonym in synonyms:
                    if synonym in query_terms:
                        matches[table] = matches.get(table, 0.0) + 0.9
                    elif synonym in self.matches:
                        for matched_table, score in self.matches[synonym].items():
                            if matched_table == table:
                                matches[table] = matches.get(table, 0.0) + score

            for term in query_terms:
                term_doc = self.nlp(term)
                for table, synonyms in self.synonyms.items():
                    for synonym in synonyms:
                        synonym_doc = self.nlp(synonym)
                        similarity = term_doc.similarity(synonym_doc)
                        if similarity > 0.8:
                            matches[table] = max(matches.get(table, 0.0), similarity)

            self.logger.debug(f"Name matches for query '{query}': {matches}")
            return matches
        except Exception as e:
            self.logger.error(f"Error in name matching: {e}")
            return {}

    def update_matches(self, query: str, tables: List[str]):
        """
        Update synonyms and matches based on user feedback.

        Args:
            query (str): Query text.
            tables (List[str]): Confirmed table names.
        """
        self.logger.debug(f"Updating synonyms for query: {query}, tables: {tables}")
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, skipping synonym update")
            return

        try:
            query_lower = query.lower()
            query_doc = self.nlp(query_lower)
            terms = [token.lemma_ for token in query_doc if not token.is_stop and token.is_alpha]

            for table in tables:
                if table not in self.synonyms:
                    self.synonyms[table] = []
                for term in terms:
                    if term not in self.synonyms[table]:
                        self.synonyms[table].append(term)
                        self.logger.debug(f"Added synonym '{term}' for {table}")
                    self.matches[term] = self.matches.get(term, {})
                    self.matches[term][table] = self.matches[term].get(table, 0.9)

            self._save_synonyms()
            self._save_matches()
            self.logger.debug(f"Updated synonyms for query: {query_lower}, terms: {terms}, tables: {tables}")
        except Exception as e:
            self.logger.error(f"Error updating synonyms: {e}")

    def _save_synonyms(self):
        """
        Save synonyms to JSON file.
        """
        try:
            with open(self.synonyms_file, 'w') as f:
                json.dump(self.synonyms, f, indent=2)
            self.logger.debug(f"Saved synonyms to {self.synonyms_file}")
        except Exception as e:
            self.logger.error(f"Error saving synonyms: {e}")

    def _save_matches(self):
        """
        Save matches to Redis.
        """
        try:
            self.redis_client.set(f"{self.db_name}:name_matches", json.dumps(self.matches))
            self.logger.debug(f"Saved matches to Redis for {self.db_name}")
        except redis.RedisError as e:
            self.logger.error(f"Redis error saving matches: {e}")