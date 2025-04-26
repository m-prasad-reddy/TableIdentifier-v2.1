"""
query_processor.py

Processes natural language queries to identify relevant database tables in the
TableIdentifier-v2.1 application. Validates queries for relevance, language, and
context, preprocesses them using SpaCy, and delegates table identification to
TableIdentifier.

Dependencies:
- spacy: For NLP processing and language detection (en_core_web_md model).
- re, logging: For query preprocessing and logging.
"""

import spacy
import re
from typing import List, Tuple
import logging

class QueryProcessor:
    """
    Processes natural language queries to identify relevant tables.

    Attributes:
        logger (logging.Logger): Logger for query processing.
        table_identifier (TableIdentifier): TableIdentifier instance for table matching.
        nlp_pipeline (NLPPipeline): NLP pipeline for additional processing (optional).
        nlp (spacy.language.Language): SpaCy NLP model (en_core_web_md).
        stopwords (set): Set of stopwords and abuse words for query validation.
    """

    def __init__(self, table_identifier, nlp_pipeline=None):
        """
        Initialize QueryProcessor with TableIdentifier and optional NLP pipeline.

        Args:
            table_identifier: TableIdentifier instance for table matching.
            nlp_pipeline: NLPPipeline instance for additional NLP processing (optional).
        """
        self.logger = logging.getLogger("query_processor")
        self.table_identifier = table_identifier
        self.nlp_pipeline = nlp_pipeline
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self.stopwords = set()
        self._load_stopwords()
        self.logger.debug("Initialized QueryProcessor")

    def _load_stopwords(self):
        """
        Load stopwords and abuse words from file.
        """
        try:
            with open(f"app-config/{self.table_identifier.db_name}/stopwords.txt", 'r') as f:
                self.stopwords = set(line.strip().lower() for line in f)
            question_words = {"what", "where", "when", "who", "how"}
            self.stopwords.difference_update(question_words)
            self.logger.debug(f"Loaded stopwords for {self.table_identifier.db_name}")
        except FileNotFoundError:
            self.logger.warning("Stopwords file not found, using default set")
            self.stopwords = set(["like", "doing", "you", "chitti", "emchestunnav"])

    def validate_query(self, query: str) -> bool:
        """
        Validate query for relevance, language, and context.

        Args:
            query (str): Natural language query.

        Returns:
            bool: True if query is valid, False otherwise.
        """
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, skipping query validation")
            return True

        query_lower = query.lower().strip()
        if len(query_lower.split()) <= 1:
            self.logger.warning(f"Single-word query: {query_lower}")
            return False

        doc = self.nlp(query_lower)
        if doc.lang_ != "en":
            self.logger.warning(f"Non-English query detected: {query}")
            return False

        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        if not tokens:
            self.logger.warning(f"No meaningful tokens in query: {query}")
            return False

        return True

    def process_query(self, query: str) -> Tuple[List[str], float]:
        """
        Process query to identify relevant tables.

        Args:
            query (str): Natural language query.

        Returns:
            Tuple[List[str], float]: List of table names and confidence score.
        """
        self.logger.debug(f"Processing query: {query}")
        if not self.validate_query(query):
            self.logger.warning(f"Invalid query: {query}")
            return [], 0.0

        try:
            preprocessed_query = self.preprocess_query(query)
            if self.nlp_pipeline and hasattr(self.nlp_pipeline, 'process'):
                preprocessed_query = self.nlp_pipeline.process(preprocessed_query)
            tables, confidence = self.table_identifier.identify_tables(preprocessed_query)
            self.logger.debug(f"Identified tables: {tables}, confidence: {confidence}")
            return tables, confidence
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return [], 0.0

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query for table identification.

        Args:
            query (str): Natural language query.

        Returns:
            str: Preprocessed query text.
        """
        self.logger.debug(f"Preprocessing query: {query}")
        if not self.nlp:
            return query.lower()

        query_lower = query.lower().strip()
        doc = self.nlp(query_lower)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        preprocessed = ' '.join(tokens)
        self.logger.debug(f"Preprocessed query: {preprocessed}")
        return preprocessed