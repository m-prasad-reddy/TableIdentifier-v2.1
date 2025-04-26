"""
table_identifier.py

Identifies database tables from natural language queries in the TableIdentifier-v2.1 application.
Uses pattern matching, name matching, and feedback to rank tables.

Dependencies:
- redis: For caching weights and matches.
- pandas: For loading training data.
- spacy: For query similarity scoring.
- logging: For logging operations.
- json: For serialization.
"""

import redis
import pandas as pd
import spacy
import logging
import json
from typing import List, Tuple, Dict

class TableIdentifier:
    """
    Identifies relevant database tables from natural language queries.

    Attributes:
        logger (logging.Logger): Logger for table identification.
        schema_manager (SchemaManager): SchemaManager instance.
        pattern_manager (PatternManager): PatternManager instance.
        name_match_manager (NameMatchManager): NameMatchManager instance.
        db_name (str): Database name.
        redis_client (redis.Redis): Redis client for caching.
        weights (Dict[str, float]): Table weights for ranking.
        training_data (List[Dict]): Training data from CSV.
        nlp (spacy.language.Language): SpaCy NLP model for similarity scoring.
    """

    def __init__(self, schema_manager, pattern_manager, name_match_manager, db_name: str):
        """
        Initialize TableIdentifier with managers and database name.

        Args:
            schema_manager: SchemaManager instance.
            pattern_manager: PatternManager instance.
            name_match_manager: NameMatchManager instance.
            db_name (str): Database name.
        """
        self.logger = logging.getLogger("table_identifier")
        self.schema_manager = schema_manager
        self.pattern_manager = pattern_manager
        self.name_match_manager = name_match_manager
        self.db_name = db_name
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.weights = {}
        self.training_data = []
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self._load_weights()
        self._load_training_data()
        self.logger.debug("Initialized TableIdentifier")

    def _load_weights(self):
        """
        Load table weights from Redis.
        """
        try:
            cached_weights = self.redis_client.get(f"{self.db_name}:weights")
            if cached_weights:
                self.weights = json.loads(cached_weights)
                self.logger.debug(f"Loaded weights from Redis for {self.db_name}")
            else:
                for table in self.schema_manager.get_all_tables():
                    self.weights[table] = 1.0
                self._save_weights()
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            for table in self.schema_manager.get_all_tables():
                self.weights[table] = 1.0

    def _save_weights(self):
        """
        Save table weights to Redis.
        """
        try:
            self.redis_client.set(f"{self.db_name}:weights", json.dumps(self.weights))
            self.logger.debug(f"Saved weights to Redis for {self.db_name}")
        except Exception as e:
            self.logger.error(f"Error saving weights: {e}")

    def _load_training_data(self):
        """
        Load training data from CSV.
        """
        try:
            df = pd.read_csv(f"app-config/{self.db_name}/db_config_trainer.csv")
            self.training_data = df.to_dict('records')
            self.logger.debug(f"Loaded {len(self.training_data)} training records")
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            self.training_data = []

    def identify_tables(self, query: str) -> Tuple[List[str], float]:
        """
        Identify relevant tables for a query.

        Args:
            query (str): Preprocessed query text.

        Returns:
            Tuple[List[str], float]: List of table names and confidence score.
        """
        self.logger.debug(f"Identifying tables for query: {query}")
        matched_tables = set()

        # Pattern-based matching
        pattern_tables = self.pattern_manager.match_pattern(query)
        matched_tables.update(pattern_tables)
        pattern_confidence = 0.9 if pattern_tables else 0.0
        self.logger.debug(f"Pattern matched tables: {pattern_tables}, confidence: {pattern_confidence}")

        # Name-based matching
        name_matches = self.name_match_manager.get_matches(query)
        name_tables = [table for table, score in name_matches.items() if score >= 0.8]
        matched_tables.update(name_tables)
        name_confidence = max([score for _, score in name_matches.items()] + [0.0])
        self.logger.debug(f"Name matched tables: {name_tables}, confidence: {name_confidence}")

        # Feedback-based matching
        feedback_tables = self._get_feedback_tables(query)
        matched_tables.update(feedback_tables)
        feedback_confidence = 0.95 if feedback_tables else 0.0
        self.logger.debug(f"Feedback matched tables: {feedback_tables}, confidence: {feedback_confidence}")

        # Combine tables
        tables = list(matched_tables)
        if not tables:
            self.logger.warning(f"No tables identified for query: {query}")
            return [], 0.0

        # Calculate confidence
        confidence = max(pattern_confidence, name_confidence, feedback_confidence) * self._calculate_weight_score(tables)
        tables = sorted(tables, key=lambda t: self.weights.get(t, 1.0), reverse=True)[:5]
        self.logger.debug(f"Final tables: {tables}, confidence: {confidence}")
        return tables, confidence

    def _get_feedback_tables(self, query: str) -> List[str]:
        """
        Get tables from similar feedback queries using SpaCy similarity.

        Args:
            query (str): Preprocessed query text.

        Returns:
            List[str]: List of table names from feedback.
        """
        feedback_tables = []
        if not self.training_data or not self.nlp:
            self.logger.debug("No training data or SpaCy model, skipping feedback matching")
            return feedback_tables

        try:
            query_doc = self.nlp(query.lower())
            for record in self.training_data:
                if record.get('query') and record.get('tables'):
                    feedback_query = record['query'].lower()
                    feedback_doc = self.nlp(feedback_query)
                    similarity = query_doc.similarity(feedback_doc)
                    if similarity > 0.85:  # High similarity threshold
                        tables = json.loads(record['tables']) if isinstance(record['tables'], str) else record['tables']
                        feedback_tables.extend(tables)
                        self.logger.debug(f"Found similar feedback (similarity {similarity:.2f}): {tables}")
            return list(set(feedback_tables))
        except Exception as e:
            self.logger.error(f"Error in feedback matching: {e}")
            return []

    def _calculate_weight_score(self, tables: List[str]) -> float:
        """
        Calculate a score based on table weights.

        Args:
            tables (List[str]): List of table names.

        Returns:
            float: Combined weight score.
        """
        if not tables:
            return 0.0
        total_weight = sum(self.weights.get(table, 1.0) for table in tables)
        return min(total_weight / len(tables), 1.0)

    def adjust_weights(self, selected_tables: List[str], all_tables: List[str]):
        """
        Adjust table weights based on user feedback.

        Args:
            selected_tables: Tables confirmed by the user.
            all_tables: All available tables.
        """
        for table in selected_tables:
            self.weights[table] = self.weights.get(table, 1.0) + 0.3
            self.logger.debug(f"Increased weight for {table} to {self.weights[table]}")
        for table in set(all_tables) - set(selected_tables):
            self.weights[table] = max(0.5, self.weights.get(table, 1.0) * 0.85)
            self.logger.debug(f"Decreased weight for {table} to {self.weights[table]}")
        # Decay unused tables
        for table in set(self.weights.keys()) - set(all_tables):
            self.weights[table] = max(0.5, self.weights.get(table, 1.0) * 0.95)
            self.logger.debug(f"Decayed weight for unused {table} to {self.weights[table]}")
        self._save_weights()

    def update_name_matches(self, query: str, selected_tables: List[str]):
        """
        Update name matches based on user feedback.

        Args:
            query: The query text.
            selected_tables: Tables confirmed by the user.
        """
        self.logger.debug(f"Updating name matches for query: {query}")
        try:
            self.name_match_manager.update_matches(query, selected_tables)
            self.redis_client.set(f"{self.db_name}:name_matches", json.dumps(self.name_match_manager.matches))
            self.logger.debug(f"Saved name matches to Redis for {self.db_name}")
        except Exception as e:
            self.logger.error(f"Error updating name matches: {e}")

    def save_model(self):
        """
        Save the model state (weights and matches).
        """
        self._save_weights()
        try:
            with open(f"models/{self.db_name}_model.json", 'w') as f:
                json.dump({"weights": self.weights}, f)
            self.logger.debug(f"Saved model to models/{self.db_name}_model.json")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")