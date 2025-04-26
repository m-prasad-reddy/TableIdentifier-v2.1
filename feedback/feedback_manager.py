"""
feedback_manager.py

Manages user feedback for query-table mappings in the TableIdentifier-v2.1 application.
Stores feedback in SQLite and caches it in Redis for fast retrieval. Supports feedback
export/import for manual interpretation and retrieval of top queries for example display.
Uses SpaCy for similarity-based feedback matching.

Dependencies:
- sqlite3: For persistent feedback storage.
- spacy: For query similarity scoring (en_core_web_md model).
- redis: For caching feedback.
- json, os, logging: For data handling and logging.
"""

import sqlite3
import spacy
import json
from typing import Dict, List
import logging
import redis
import os

class FeedbackManager:
    """
    Manages feedback for query-table mappings using Redis and SQLite.

    Attributes:
        logger (logging.Logger): Logger for feedback operations.
        db_name (str): Database name.
        redis_client (redis.Redis): Redis client for caching.
        feedback_key (str): Redis key for feedback cache.
        db_path (str): Path to SQLite feedback database.
        export_path (str): Path for feedback export JSON.
        nlp (spacy.language.Language): SpaCy NLP model (en_core_web_md).
    """

    def __init__(self, db_name: str, redis_client: redis.Redis):
        """
        Initialize FeedbackManager with database name and Redis client.

        Args:
            db_name (str): Database name.
            redis_client (redis.Redis): Redis client.
        """
        self.logger = logging.getLogger("feedback_manager")
        self.db_name = db_name
        self.redis_client = redis_client
        self.feedback_key = f"{db_name}:feedback"
        self.db_path = f"feedback_cache/{db_name}/feedback.db"
        self.export_path = f"feedback_cache/{db_name}/feedback_export.json"
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self._initialize_db()

    def _initialize_db(self):
        """
        Initialize SQLite database for feedback storage.
        """
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    query TEXT,
                    tables TEXT,
                    weight REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_counts (
                    query TEXT PRIMARY KEY,
                    count INTEGER
                )
            """)
            conn.commit()
            conn.close()
            self.logger.debug(f"Initialized feedback database at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error initializing feedback database: {e}")

    def get_similar_feedback(self, query: str) -> Dict:
        """
        Retrieve feedback for similar queries.

        Args:
            query (str): Query text.

        Returns:
            Dict: Feedback with query, tables, and weight, or empty dict if none found.
        """
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, skipping feedback")
            return {}

        try:
            # Check Redis cache
            feedback_data = self.redis_client.get(self.feedback_key)
            if feedback_data:
                feedback_list = json.loads(feedback_data)
                doc = self.nlp(query.lower())
                max_similarity = 0.0
                best_feedback = {}
                for entry in feedback_list:
                    entry_doc = self.nlp(entry['query'].lower())
                    similarity = doc.similarity(entry_doc)
                    if similarity > max_similarity and similarity > 0.8:
                        max_similarity = similarity
                        best_feedback = entry
                if best_feedback:
                    self.logger.debug(f"Found similar feedback in Redis: {best_feedback}")
                    return best_feedback

            # Fallback to SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT query, tables, weight FROM feedback")
            feedback_list = cursor.fetchall()
            conn.close()

            doc = self.nlp(query.lower())
            max_similarity = 0.0
            best_feedback = {}
            for entry_query, entry_tables, weight in feedback_list:
                entry_doc = self.nlp(entry_query.lower())
                similarity = doc.similarity(entry_doc)
                if similarity > max_similarity and similarity > 0.8:
                    max_similarity = similarity
                    best_feedback = {"query": entry_query, "tables": json.loads(entry_tables), "weight": weight}

            if best_feedback:
                feedback_list = [best_feedback]
                self.redis_client.set(self.feedback_key, json.dumps(feedback_list))
                self.logger.debug(f"Found similar feedback in SQLite: {best_feedback}")
                return best_feedback
            return {}
        except Exception as e:
            self.logger.error(f"Error retrieving feedback: {e}")
            return {}

    def store_feedback(self, query: str, tables: List[str], schema_dict: Dict = None, weight: float = 1.0):
        """
        Store feedback in SQLite and Redis.

        Args:
            query (str): Query text.
            tables (List[str]): Confirmed tables.
            schema_dict (Dict): Schema dictionary (optional).
            weight (float): Feedback weight (default: 1.0).
        """
        try:
            # Store in SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO feedback (query, tables, weight) VALUES (?, ?, ?)",
                           (query, json.dumps(tables), weight))
            cursor.execute("INSERT OR REPLACE INTO query_counts (query, count) VALUES (?, ?)",
                           (query, self.get_query_count(query) + 1))
            conn.commit()
            conn.close()

            # Update Redis cache
            feedback_data = self.redis_client.get(self.feedback_key)
            feedback_list = json.loads(feedback_data) if feedback_data else []
            feedback_list.append({"query": query, "tables": tables, "weight": weight})
            self.redis_client.set(self.feedback_key, json.dumps(feedback_list))
            self.logger.debug(f"Stored feedback for query: {query}, tables: {tables}")
        except Exception as e:
            self.logger.error(f"Error storing feedback: {e}")

    def get_query_count(self, query: str) -> int:
        """
        Get the count of how many times a query was used.

        Args:
            query (str): Query text.

        Returns:
            int: Number of times the query was used.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT count FROM query_counts WHERE query = ?", (query,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error retrieving query count: {e}")
            return 0

    def get_top_queries(self, limit: int = 5) -> List[Dict]:
        """
        Retrieve the top queries by frequency or weight.

        Args:
            limit (int): Maximum number of queries to return (default: 5).

        Returns:
            List[Dict]: List of dictionaries with query, tables, weight, and count.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.query, f.tables, f.weight, COALESCE(qc.count, 0) as count
                FROM feedback f
                LEFT JOIN query_counts qc ON f.query = qc.query
                ORDER BY qc.count DESC, f.weight DESC
                LIMIT ?
            """, (limit,))
            top_queries = [
                {"query": query, "tables": json.loads(tables), "weight": weight, "count": count}
                for query, tables, weight, count in cursor.fetchall()
            ]
            conn.close()
            self.logger.debug(f"Retrieved {len(top_queries)} top queries")
            return top_queries
        except Exception as e:
            self.logger.error(f"Error retrieving top queries: {e}")
            return []

    def export_feedback(self, export_path: str = None):
        """
        Export feedback data to a JSON file for manual interpretation.

        Args:
            export_path (str): Path for export file (default: feedback_cache/{db_name}/feedback_export.json).
        """
        if not export_path:
            export_path = self.export_path

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT query, tables, weight FROM feedback")
            feedback_list = [
                {"query": query, "tables": json.loads(tables), "weight": weight}
                for query, tables, weight in cursor.fetchall()
            ]
            cursor.execute("SELECT query, count FROM query_counts")
            query_counts = {query: count for query, count in cursor.fetchall()}
            conn.close()

            export_data = {
                "feedback": feedback_list,
                "query_counts": query_counts
            }

            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Exported feedback to {export_path}")
        except Exception as e:
            self.logger.error(f"Error exporting feedback: {e}")

    def import_feedback(self, import_path: str = None):
        """
        Import feedback data from a JSON file and merge with existing data.

        Args:
            import_path (str): Path for import file (default: feedback_cache/{db_name}/feedback_export.json).
        """
        if not import_path:
            import_path = self.export_path

        try:
            if not os.path.exists(import_path):
                self.logger.warning(f"Import file not found: {import_path}")
                return

            with open(import_path, 'r') as f:
                import_data = json.load(f)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Import feedback
            for entry in import_data.get("feedback", []):
                query = entry.get("query")
                tables = entry.get("tables")
                weight = entry.get("weight", 1.0)
                if query and tables:
                    cursor.execute("INSERT OR REPLACE INTO feedback (query, tables, weight) VALUES (?, ?, ?)",
                                   (query, json.dumps(tables), weight))

            # Import query counts
            for query, count in import_data.get("query_counts", {}).items():
                cursor.execute("INSERT OR REPLACE INTO query_counts (query, count) VALUES (?, ?)",
                               (query, count))

            conn.commit()
            conn.close()

            # Update Redis cache
            feedback_list = [
                {"query": entry["query"], "tables": entry["tables"], "weight": entry["weight"]}
                for entry in import_data.get("feedback", [])
            ]
            self.redis_client.set(self.feedback_key, json.dumps(feedback_list))
            self.logger.info(f"Imported feedback from {import_path}")
        except Exception as e:
            self.logger.error(f"Error importing feedback: {e}")

    def clear_feedback(self):
        """
        Clear all feedback data from SQLite and Redis.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feedback")
            cursor.execute("DELETE FROM query_counts")
            conn.commit()
            conn.close()

            self.redis_client.delete(self.feedback_key)
            self.logger.info("Cleared all feedback data")
        except Exception as e:
            self.logger.error(f"Error clearing feedback: {e}")