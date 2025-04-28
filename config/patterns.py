"""
patterns.py

Manages query patterns for table identification in the TableIdentifier-v2.1 application.
Dynamically generates SpaCy-compatible patterns from schema metadata and caches them in Redis.
Matches query intent to tables using keywords, entities, and foreign keys, updating patterns based on feedback.

Dependencies:
- spacy: For NLP processing (en_core_web_md model).
- redis: For caching patterns and entity mappings.
- json, os, logging: For file handling and logging.
"""

import json
import os
import spacy
from typing import Dict, List
import redis
import logging

class PatternManager:
    """
    Manages query patterns for table identification using Redis caching.

    Generates SpaCy-compatible patterns from schema metadata, matches queries to tables,
    and updates patterns based on user feedback with intent-specific keyword mapping.

    Attributes:
        logger (logging.Logger): Logger for pattern operations.
        schema_dict (Dict): Database schema metadata (tables, columns, foreign keys).
        db_name (str): Database name (e.g., BikeStores).
        redis_client (redis.Redis): Redis client for caching.
        patterns_key (str): Redis key for SpaCy-compatible patterns.
        entities_key (str): Redis key for entity mappings.
        nlp (spacy.language.Language): SpaCy NLP model (en_core_web_md).
        patterns (Dict[str, List[List[Dict]]]): SpaCy-compatible patterns for tables.
        entity_mappings (Dict[str, List[str]]): Intent-to-column mappings.
    """

    def __init__(self, schema_dict: Dict, db_name: str, redis_client: redis.Redis):
        """
        Initialize PatternManager with schema, database name, and Redis client.

        Args:
            schema_dict (Dict): Schema dictionary with tables, columns, and foreign keys.
            db_name (str): Database name.
            redis_client (redis.Redis): Redis client for caching.
        """
        self.logger = logging.getLogger("patterns")
        self.schema_dict = schema_dict
        self.db_name = db_name
        self.redis_client = redis_client
        self.patterns_key = f"{db_name}:patterns"
        self.entities_key = f"{db_name}:entities"
        os.makedirs(f"app-config/{db_name}", exist_ok=True)
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self.patterns = self._load_patterns()
        self.entity_mappings = self._load_entity_mappings()
        self.logger.debug(f"Initialized PatternManager for {db_name} with {len(self.patterns)} patterns")

    def _load_patterns(self) -> Dict[str, List[List[Dict]]]:
        """
        Load SpaCy-compatible patterns from Redis or initialize from schema.

        Returns:
            Dict[str, List[List[Dict]]]: Patterns mapping tables to SpaCy patterns.
        """
        try:
            patterns_data = self.redis_client.get(self.patterns_key)
            if patterns_data:
                patterns = json.loads(patterns_data)
                self.logger.debug(f"Loaded patterns from Redis for {self.db_name}")
                return patterns
            self.logger.debug(f"No patterns in Redis for {self.db_name}, initializing")
            self._initialize_patterns()
            patterns_data = self.redis_client.get(self.patterns_key)
            return json.loads(patterns_data) if patterns_data else {}
        except redis.RedisError as e:
            self.logger.error(f"Redis error loading patterns: {e}, initializing default")
            self._initialize_patterns()
            return self.patterns
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for patterns: {e}, initializing default")
            self._initialize_patterns()
            return self.patterns

    def _load_entity_mappings(self) -> Dict[str, List[str]]:
        """
        Load entity mappings from Redis or initialize.

        Returns:
            Dict[str, List[str]]: Intent-to-column mappings.
        """
        try:
            entities_data = self.redis_client.get(self.entities_key)
            if entities_data:
                entities = json.loads(entities_data)
                self.logger.debug(f"Loaded entities from Redis for {self.db_name}")
                return entities
            self.logger.debug(f"No entities in Redis for {self.db_name}, initializing")
            self._initialize_entity_mappings()
            entities_data = self.redis_client.get(self.entities_key)
            return json.loads(entities_data) if entities_data else {}
        except redis.RedisError as e:
            self.logger.error(f"Redis error loading entities: {e}, using default")
            return self._get_default_entity_mappings()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for entities: {e}, using default")
            return self._get_default_entity_mappings()

    def _get_default_entity_mappings(self) -> Dict[str, List[str]]:
        """
        Provide default entity mappings when Redis fails.

        Returns:
            Dict[str, List[str]]: Default intent-to-column mappings.
        """
        return {
            "name": ["name", "first_name", "last_name", "fname", "lname"],
            "details": ["address", "city", "state", "zip_code", "store_name"],
            "stock": ["quantity", "stock", "inventory"],
            "category": ["category", "category_name"],
            "order": ["order", "order_id", "sales", "purchase"],
            "brand": ["brand", "brand_name"]
        }

    def _initialize_patterns(self) -> None:
        """
        Initialize SpaCy-compatible patterns from schema metadata and store in Redis.
        """
        self.patterns = {}
        for schema in self.schema_dict['tables']:
            for table in self.schema_dict['tables'][schema]:
                full_table = f"{schema}.{table}"
                patterns = []

                # Basic table and schema patterns
                patterns.append([{"LOWER": table.lower()}])
                patterns.append([{"LOWER": schema.lower()}])

                # Column-based patterns
                for col_name in self.schema_dict['columns'][schema][table]:
                    col_clean = col_name.lower().replace('_', ' ')
                    patterns.append([{"LOWER": col_clean}])

                # Context-specific patterns
                if any(col in self.schema_dict['columns'][schema][table] for col in ["first_name", "last_name"]):
                    patterns.append([{"LOWER": "customer"}, {"LOWER": "name"}])
                    patterns.append([{"LOWER": "employee"}, {"LOWER": "name"}])
                if any(col in self.schema_dict['columns'][schema][table] for col in ["quantity", "stock"]):
                    patterns.append([{"LOWER": "stock"}, {"LOWER": "availability"}])
                if "category_name" in self.schema_dict['columns'][schema][table]:
                    patterns.append([{"LOWER": "product"}, {"LOWER": "category"}])
                if "order_id" in self.schema_dict['columns'][schema][table]:
                    patterns.append([{"LOWER": "order"}, {"LOWER": "details"}])
                if "store_name" in self.schema_dict['columns'][schema][table]:
                    patterns.append([{"LOWER": "store"}, {"LOWER": "details"}])
                    patterns.append([{"LOWER": "stores"}])

                self.patterns[full_table] = patterns
                self.logger.debug(f"Generated {len(patterns)} patterns for {full_table}")

        try:
            self.redis_client.set(self.patterns_key, json.dumps(self.patterns))
            self.logger.debug(f"Initialized patterns in Redis for {self.db_name}")
        except redis.RedisError as e:
            self.logger.error(f"Redis error saving patterns: {e}")

    def _initialize_entity_mappings(self) -> None:
        """
        Initialize entity mappings for intent matching and store in Redis.
        """
        entities = self._get_default_entity_mappings()
        for schema in self.schema_dict['columns']:
            for table in self.schema_dict['columns'][schema]:
                for col_name in self.schema_dict['columns'][schema][table]:
                    col_lower = col_name.lower()
                    if "name" in col_lower:
                        entities["name"].append(col_lower)
                    if any(term in col_lower for term in ["address", "city", "state", "zip"]):
                        entities["details"].append(col_lower)
                    if any(term in col_lower for term in ["quantity", "stock", "inventory"]):
                        entities["stock"].append(col_lower)
                    if "category" in col_lower:
                        entities["category"].append(col_lower)
                    if any(term in col_lower for term in ["order", "sales", "purchase"]):
                        entities["order"].append(col_lower)
                    if "brand" in col_lower:
                        entities["brand"].append(col_lower)
        for intent in entities:
            entities[intent] = list(set(entities[intent]))
        try:
            self.redis_client.set(self.entities_key, json.dumps(entities))
            self.logger.debug(f"Initialized entities in Redis for {self.db_name}")
        except redis.RedisError as e:
            self.logger.error(f"Redis error saving entities: {e}")
            self.entity_mappings = entities

    def match_pattern(self, query: str) -> List[str]:
        """
        Match query against patterns, entities, and foreign keys with higher specificity.

        Args:
            query (str): Natural language query.

        Returns:
            List[str]: Matching table names (schema.table).
        """
        self.logger.debug(f"Matching patterns for query: {query}")
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, using keyword matching")
            return self._keyword_fallback(query)

        try:
            query_lower = query.lower()
            doc = self.nlp(query_lower)
            matches = set()
            query_tokens = {token.text for token in doc if not token.is_stop and token.is_alpha}

            # Pattern-based matching with exact matches
            for table, patterns in self.patterns.items():
                for pattern in patterns:
                    pattern_str = ' '.join(token["LOWER"] for token in pattern)
                    if pattern_str in query_lower:
                        matches.add(table)
                        continue
                    # Check if all pattern tokens are in query
                    pattern_tokens = {token["LOWER"] for token in pattern}
                    if pattern_tokens.issubset(query_tokens):
                        matches.add(table)

            # Entity-based matching with column validation
            for ent in doc.ents:
                ent_type = ent.label_
                if ent_type in ["DATE", "GPE"]:
                    for schema in self.schema_dict['columns']:
                        for table in self.schema_dict['columns'][schema]:
                            full_table = f"{schema}.{table}"
                            for col_name, col_info in self.schema_dict['columns'][schema][table].items():
                                if ent_type == "DATE" and col_info['type'].lower() in ['date', 'datetime', 'timestamp']:
                                    matches.add(full_table)
                                if ent_type == "GPE" and any(term in col_name.lower() for term in ["city", "state", "country"]):
                                    matches.add(full_table)

            # Intent-based matching with column relevance
            for intent, keywords in self.entity_mappings.items():
                matched_tables = set()
                for keyword in keywords:
                    if keyword in query_lower:
                        for schema in self.schema_dict['columns']:
                            for table in self.schema_dict['columns'][schema]:
                                full_table = f"{schema}.{table}"
                                for col_name in self.schema_dict['columns'][schema][table]:
                                    if keyword in col_name.lower():
                                        matched_tables.add(full_table)
                matches.update(matched_tables)

            # Foreign key relationships
            fk_matches = set()
            for schema in self.schema_dict.get('foreign_keys', {}):
                for table in self.schema_dict['foreign_keys'][schema]:
                    full_table = f"{schema}.{table}"
                    if full_table in matches:
                        for fk in self.schema_dict['foreign_keys'][schema][table]:
                            ref_table = fk['referenced_table']
                            fk_matches.add(ref_table)
            matches.update(fk_matches)

            matches = list(matches)
            self.logger.debug(f"Pattern matches: {matches}")
            return matches
        except Exception as e:
            self.logger.error(f"Error in pattern matching: {e}")
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> List[str]:
        """
        Fallback keyword matching when SpaCy is unavailable.

        Args:
            query (str): Natural language query.

        Returns:
            List[str]: Matching table names.
        """
        query_lower = query.lower()
        matches = set()
        for schema in self.schema_dict['tables']:
            for table in self.schema_dict['tables'][schema]:
                full_table = f"{schema}.{table}"
                if table.lower() in query_lower or schema.lower() in query_lower:
                    matches.add(full_table)
                for col_name in self.schema_dict['columns'][schema][table]:
                    if col_name.lower().replace('_', ' ') in query_lower:
                        matches.add(full_table)
        matches = list(matches)
        self.logger.debug(f"Fallback matches: {matches}")
        return matches

    def get_patterns(self) -> Dict[str, List[List[Dict]]]:
        """
        Return SpaCy-compatible patterns for NLPPipeline.

        Returns:
            Dict[str, List[List[Dict]]]: Patterns mapping tables to SpaCy patterns.
        """
        return self.patterns

    def update_patterns(self, query: str, tables: List[str]) -> None:
        """
        Update patterns and entity mappings based on user feedback with intent-specific mapping.

        Adds query keywords to patterns and intents only if relevant to table columns.

        Args:
            query (str): Query text.
            tables (List[str]): Confirmed table names.
        """
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, skipping pattern update")
            return

        try:
            query_lower = query.lower()
            doc = self.nlp(query_lower)
            keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

            for table in tables:
                if table not in self.patterns:
                    self.patterns[table] = []
                schema, table_name = table.split('.')
                relevant_columns = self.schema_dict['columns'].get(schema, {}).get(table_name, {})

                for keyword in keywords:
                    # Add single-keyword pattern if relevant
                    new_pattern = [{"LOWER": keyword}]
                    is_relevant = False
                    for col_name in relevant_columns:
                        if keyword in col_name.lower().replace('_', ' '):
                            is_relevant = True
                            break
                    if is_relevant and new_pattern not in self.patterns[table]:
                        self.patterns[table].append(new_pattern)
                        self.logger.debug(f"Added pattern {new_pattern} for {table}")

                    # Update entity mappings for relevant intents
                    for intent, keywords_list in self.entity_mappings.items():
                        if intent == "name" and any(col in relevant_columns for col in ["first_name", "last_name", "name"]):
                            if keyword not in keywords_list:
                                keywords_list.append(keyword)
                                self.logger.debug(f"Added keyword '{keyword}' to intent '{intent}' for {table}")
                        elif intent == "details" and any(col in relevant_columns for col in ["address", "city", "state", "zip_code", "store_name"]):
                            if keyword not in keywords_list:
                                keywords_list.append(keyword)
                                self.logger.debug(f"Added keyword '{keyword}' to intent '{intent}' for {table}")
                        elif intent == "stock" and any(col in relevant_columns for col in ["quantity", "stock", "inventory"]):
                            if keyword not in keywords_list:
                                keywords_list.append(keyword)
                                self.logger.debug(f"Added keyword '{keyword}' to intent '{intent}' for {table}")
                        elif intent == "category" and "category_name" in relevant_columns:
                            if keyword not in keywords_list:
                                keywords_list.append(keyword)
                                self.logger.debug(f"Added keyword '{keyword}' to intent '{intent}' for {table}")
                        elif intent == "order" and any(col in relevant_columns for col in ["order_id", "sales"]):
                            if keyword not in keywords_list:
                                keywords_list.append(keyword)
                                self.logger.debug(f"Added keyword '{keyword}' to intent '{intent}' for {table}")
                        elif intent == "brand" and "brand_name" in relevant_columns:
                            if keyword not in keywords_list:
                                keywords_list.append(keyword)
                                self.logger.debug(f"Added keyword '{keyword}' to intent '{intent}' for {table}")

            # Save updated patterns and entities to Redis
            try:
                self.redis_client.set(self.patterns_key, json.dumps(self.patterns))
                self.redis_client.set(self.entities_key, json.dumps(self.entity_mappings))
                self.logger.debug(f"Updated patterns and entities in Redis for {self.db_name}")
            except redis.RedisError as e:
                self.logger.error(f"Redis error saving updated patterns/entities: {e}")
        except Exception as e:
            self.logger.error(f"Error updating patterns: {e}")