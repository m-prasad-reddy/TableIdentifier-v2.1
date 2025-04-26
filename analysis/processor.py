"""
processor.py

Provides the NLPPipeline class for preprocessing natural language queries in the
TableIdentifier-v2.1 application. Applies pattern-based entity recognition and token
normalization using SpaCy.

Dependencies:
- spacy: For NLP processing (en_core_web_md model).
- logging: For logging operations.
"""

import logging
import spacy
from typing import Dict, List

class NLPPipeline:
    """
    Preprocesses natural language queries for table identification.

    Attributes:
        logger (logging.Logger): Logger for NLP operations.
        pattern_manager (PatternManager): PatternManager instance for entity patterns.
        db_name (str): Database name.
        nlp (spacy.language.Language): SpaCy NLP model (en_core_web_md).
        matcher (spacy.matcher.Matcher): SpaCy matcher for pattern-based entity recognition.
    """

    def __init__(self, pattern_manager, db_name: str):
        """
        Initialize NLPPipeline with PatternManager and database name.

        Args:
            pattern_manager: PatternManager instance.
            db_name (str): Database name.
        """
        self.logger = logging.getLogger("nlp_pipeline")
        self.pattern_manager = pattern_manager
        self.db_name = db_name
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self.matcher = None
        self._initialize_matcher()
        self.logger.debug("Initialized NLPPipeline")

    def _initialize_matcher(self):
        """
        Initialize SpaCy matcher with entity patterns from PatternManager.
        """
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, skipping matcher initialization")
            return

        try:
            self.matcher = spacy.matcher.Matcher(self.nlp.vocab)
            patterns = self.pattern_manager.get_patterns()
            for table, pattern_list in patterns.items():
                self.matcher.add(table, pattern_list)
                self.logger.debug(f"Added patterns for table '{table}'")
            self.logger.debug("Patterns loaded")
        except Exception as e:
            self.logger.error(f"Error initializing matcher: {e}")
            self.matcher = None

    def process(self, query: str) -> str:
        """
        Preprocess a query by applying entity recognition and token normalization.

        Args:
            query (str): The query text.

        Returns:
            str: Preprocessed query text with entities replaced by table names.
        """
        self.logger.debug(f"Processing query: {query}")
        if not self.nlp or not self.matcher:
            self.logger.warning("SpaCy model or matcher not initialized, returning original query")
            return query.lower()

        try:
            doc = self.nlp(query.lower())
            matches = self.matcher(doc)
            entities = []
            for match_id, start, end in matches:
                table = self.nlp.vocab.strings[match_id]
                entities.append((table, doc[start:end].text, start, end))

            # Replace matched entities with table names
            tokens = []
            last_end = 0
            for table, text, start, end in sorted(entities, key=lambda x: x[2]):
                tokens.extend([token.lemma_ for token in doc[last_end:start] if not token.is_stop and token.is_alpha])
                tokens.append(table)
                last_end = end
            tokens.extend([token.lemma_ for token in doc[last_end:] if not token.is_stop and token.is_alpha])

            preprocessed = ' '.join(tokens)
            self.logger.debug(f"Preprocessed query: {preprocessed}")
            return preprocessed
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return query.lower()