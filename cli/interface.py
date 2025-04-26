"""
interface.py

Command-line interface for the TableIdentifier-v2.1 application. Provides a menu-driven
interface for connecting to databases, processing queries, reloading configurations,
and managing feedback.

Dependencies:
- logging: For logging user interactions and errors.
- spacy: For semantic query validation (en_core_web_md model).
- os: For file path handling.
- typing: For type hints.
"""

import logging
import spacy
import os
from typing import List, Dict

class DatabaseAnalyzerCLI:
    """
    Command-line interface for interacting with the DatabaseAnalyzer.

    Attributes:
        logger (logging.Logger): Logger for CLI operations.
        analyzer (DatabaseAnalyzer): The DatabaseAnalyzer instance.
        nlp (spacy.language.Language): SpaCy NLP model (en_core_web_md).
        example_queries (List[str]): Fallback example queries.
        irrelevant_keywords (set): Keywords indicating non-relevant queries.
    """

    def __init__(self, analyzer):
        """
        Initialize with analyzer instance.

        Args:
            analyzer: DatabaseAnalyzer instance.
        """
        self.logger = logging.getLogger("interface")
        self.analyzer = analyzer
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            self.logger.error(f"Failed to load SpaCy model: {e}")
            self.nlp = None
        self.example_queries = [
            "Show me all stores with store names",
            "List all products with prices",
            "Show customers from a specific city",
            "Find orders placed in the last month",
            "Show stock availability for all products"
        ]
        self.irrelevant_keywords = {
            "chitti", "emchestunnav", "bagunnava", "babu", "like", "you", "doing",
            "sonia", "kill", "love", "hate", "friend", "hello", "hi", "how are"
        }
        self.logger.debug("Initialized DatabaseAnalyzerCLI")

    def run(self):
        """
        Run the main CLI loop with menu options.
        """
        db_name = self.analyzer.current_config.get('database', 'Database') if self.analyzer.current_config else 'Database'
        self.logger.info(f"Starting {db_name} Schema Analyzer")
        print(f"\n=== {db_name} Schema Analyzer ===\n")
        while True:
            print("Main Menu:")
            print("1. Connect to Database")
            print("2. Query Mode")
            print("3. Reload Configurations")
            print("4. Manage Feedback")
            print("5. Exit")
            choice = input("Select option: ").strip()

            if choice == "1":
                self._handle_connection()
                db_name = self.analyzer.current_config.get('database', 'Database') if self.analyzer.current_config else 'Database'
                print(f"\n=== {db_name} Schema Analyzer ===\n")
            elif choice == "2":
                self._query_mode()
            elif choice == "3":
                self._reload_configurations()
            elif choice == "4":
                self._manage_feedback()
            elif choice == "5":
                self.logger.info("Exiting application")
                print("Exiting...")
                break
            else:
                print("Invalid choice")

    def _handle_connection(self):
        """
        Handle database connection process.
        """
        config_path = input("Config path [default: app-config/database_configurations.json]: ").strip()
        if not config_path:
            config_path = "app-config/database_configurations.json"

        try:
            configs = self.analyzer.load_configs(config_path)
            if not configs:
                print("No valid configurations found.")
                return
            self._select_configuration(configs)
            if self.analyzer.connect_to_database():
                self.logger.info("Successfully connected to database")
                print("Successfully connected!")
            else:
                self.logger.error("Connection failed: Unable to establish database connection")
                print("Connection failed: Unable to establish database connection")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            print(f"Connection failed: {e}")

    def _select_configuration(self, configs: Dict):
        """
        Select a database configuration from available options.

        Args:
            configs: Dictionary of available configurations.
        """
        print("\nAvailable Configurations:")
        for i, name in enumerate(configs.keys(), 1):
            print(f"{i}. {name}")
        print(f"{len(configs) + 1}. Cancel")

        while True:
            choice = input("Select configuration: ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(configs):
                    config = list(configs.values())[index]
                    self.analyzer.set_current_config(config)
                    self.logger.debug(f"Selected configuration: {config.get('database')}")
                    return
                elif index == len(configs):
                    self.logger.debug("Configuration selection cancelled")
                    return
            print("Invalid selection")

    def _validate_query(self, query: str) -> bool:
        """
        Validate if a query is meaningful and relevant to the database schema.

        Args:
            query: The query to validate.

        Returns:
            bool: True if the query is valid, False otherwise.
        """
        self.logger.debug(f"Validating query: {query}")
        if not query or query.isspace():
            self.logger.warning("Empty query")
            return False

        query_lower = query.lower().strip()
        tokens = query_lower.split()
        if len(tokens) <= 1:
            self.logger.warning(f"Single-word query: {query}")
            return False
        if query_lower.isdigit() or query_lower.startswith('+'):
            self.logger.warning(f"Numeric or invalid query: {query}")
            return False

        # Check for irrelevant keywords
        if any(keyword in query_lower for keyword in self.irrelevant_keywords):
            self.logger.warning(f"Irrelevant query detected: {query}")
            return False

        if self.nlp:
            doc = self.nlp(query_lower)
            # Require at least one relevant keyword related to database
            relevant_keywords = {
                "product", "products", "stock", "stocks", "store", "stores", "customer",
                "customers", "order", "orders", "staff", "employee", "employees", "category",
                "categories", "brand", "brands", "sales", "inventory", "shop", "shops"
            }
            has_relevant_token = any(token.text in relevant_keywords for token in doc)
            if not has_relevant_token:
                self.logger.warning(f"Query lacks relevant database terms: {query}")
                return False

            # Require noun chunk or verb for meaningful structure
            has_noun_chunk = any(chunk for chunk in doc.noun_chunks)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            if not (has_noun_chunk or has_verb):
                self.logger.warning(f"Query lacks meaningful structure: {query}")
                return False

            # Ensure query is mostly English
            english_tokens = sum(1 for token in doc if token.is_alpha and token.lang_ == "en")
            if english_tokens < len([token for token in doc if token.is_alpha]) * 0.7:
                self.logger.warning(f"Query may not be in English: {query}")
                return False

        return True

    def _display_example_queries(self):
        """
        Display fallback example queries to guide the user.
        """
        print("\nPlease enter a meaningful query in English. Examples:")
        for i, example in enumerate(self.example_queries, 1):
            print(f"{i}. {example}")

    def _query_mode(self):
        """
        Enter query mode to process natural language queries.
        """
        if not self.analyzer.is_connected():
            self.logger.error("Not connected to database")
            print("Not connected to database!")
            return

        if not self.analyzer.feedback_manager:
            self.logger.error("Feedback manager not initialized")
            print("Feedback manager not initialized. Please reconnect to the database.")
            return

        try:
            example_queries = self.analyzer.feedback_manager.get_top_queries(limit=3)
            if example_queries:
                print("\nExample Queries:")
                # Deduplicate queries
                seen = set()
                unique_queries = []
                for entry in example_queries:
                    query_key = (entry['query'], tuple(sorted(entry['tables'])))
                    if query_key not in seen:
                        seen.add(query_key)
                        unique_queries.append(entry)
                for i, entry in enumerate(unique_queries[:3], 1):
                    query = entry['query']
                    tables = ', '.join(entry['tables'])
                    count = entry['count']
                    print(f"{i}. {query} (tables: {tables}, used {count} times)")
            else:
                self._display_example_queries()
        except Exception as e:
            self.logger.error(f"Error loading example queries: {e}")
            print(f"Error loading example queries: {e}")
            self._display_example_queries()

        while True:
            query = input("\nEnter query (or 'back'): ").strip()
            if query.lower() == 'back':
                self.logger.debug("Exiting query mode")
                return

            if not self._validate_query(query):
                self.logger.warning(f"Invalid query: {query}")
                print("Please enter a meaningful query in English.")
                self._display_example_queries()
                continue

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results, confidence = self.analyzer.process_query(query)
                    if results is None:
                        self.logger.error("Unable to process query")
                        print("Unable to process query. Please try again or reconnect.")
                        continue

                    if confidence >= 0.7 and results:
                        self.logger.info(f"Suggested tables for query '{query}': {results}, confidence: {confidence}")
                        print("\nSuggested Tables:")
                        for i, table in enumerate(results[:5], 1):
                            print(f"{i}. {table}")
                        self._handle_feedback(query, results)
                        break
                    else:
                        self.logger.warning(f"Low confidence ({confidence}) for query '{query}'")
                        print("\nLow confidence. Please select tables manually:")
                        self._manual_table_selection(query)
                        break
                except Exception as e:
                    self.logger.error(f"Error processing query: {e}")
                    print(f"Error processing query: {e}")
                    if attempt < max_retries - 1:
                        self.logger.debug(f"Retrying query processing (attempt {attempt + 2}/{max_retries})")
                        continue
                    print("Max retries reached. Please try again or reconnect.")
                    break

    def _handle_feedback(self, query: str, results: List[str]):
        """
        Handle user feedback for suggested tables.

        Args:
            query: The query.
            results: Suggested tables.
        """
        self.logger.debug(f"Handling feedback for query: {query}")
        while True:
            feedback = input("\nCorrect? (Y/N): ").strip().lower()
            if feedback in ('y', 'n'):
                break
            print("Please enter 'Y' or 'N'.")

        if feedback == 'y':
            self.logger.info(f"Confirmed tables for query '{query}': {results}")
            self.analyzer.confirm_tables(query, results)
        elif feedback == 'n':
            self.logger.info(f"User rejected tables for query '{query}': {results}")
            correct_tables = self._get_manual_tables()
            if correct_tables:
                self.logger.info(f"Updated feedback with tables: {correct_tables}")
                self.analyzer.update_feedback(query, correct_tables)

    def _get_manual_tables(self) -> List[str]:
        """
        Get manual table selections from the user.

        Returns:
            List of selected tables.
        """
        print("Available Tables:")
        tables = self.analyzer.get_all_tables()
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")

        selection = input("Enter table numbers or names (comma-separated, e.g., '6' or 'sales.stores'): ").strip()
        if not selection:
            self.logger.debug("No manual tables selected")
            return []

        selected = []
        items = [s.strip() for s in selection.split(',')]
        for item in items:
            if item.isdigit():
                try:
                    index = int(item) - 1
                    if 0 <= index < len(tables):
                        selected.append(tables[index])
                except (IndexError, ValueError):
                    continue
            elif '.' in item and item in tables:
                selected.append(item)

        if not selected:
            self.logger.warning("Invalid manual table selection")
            print("Invalid selection, please try again")
        else:
            self.logger.debug(f"Selected manual tables: {selected}")
        return selected

    def _manual_table_selection(self, query: str):
        """
        Handle manual table selection for a query.

        Args:
            query F: The query.
        """
        selected_tables = self._get_manual_tables()
        if selected_tables:
            self.logger.info(f"Manually selected tables for query '{query}': {selected_tables}")
            self.analyzer.update_feedback(query, selected_tables)

    def _reload_configurations(self):
        """
        Reload all configurations and caches.
        """
        try:
            if self.analyzer.reload_all_configurations():
                self.logger.info("Successfully reloaded configurations")
                print("Successfully reloaded configurations")
            else:
                self.logger.error("Reload failed: Unable to reload configurations")
                print("Reload failed: Unable to reload configurations")
        except Exception as e:
            self.logger.error(f"Reload failed: {e}")
            print(f"Reload failed: {e}")

    def _manage_feedback(self):
        """
        Manage feedback operations (export, import, clear).
        """
        if not self.analyzer.is_connected():
            self.logger.error("Not connected to database for feedback management")
            print("Please connect to a database to manage feedback.")
            return

        if not self.analyzer.feedback_manager:
            self.logger.error("Feedback manager not initialized")
            print("Feedback manager not initialized. Please reconnect to the database.")
            return

        while True:
            print("\nFeedback Management:")
            print("1. Export feedback")
            print("2. Import feedback")
            print("3. Clear feedback")
            print("4. Back")
            choice = input("Select option: ").strip()

            if choice == "1":
                self._export_feedback()
            elif choice == "2":
                self._import_feedback()
            elif choice == "3":
                try:
                    self.analyzer.feedback_manager.clear_feedback()
                    self.logger.info("Feedback cleared")
                    print("Feedback cleared")
                except Exception as e:
                    self.logger.error(f"Error clearing feedback: {e}")
                    print(f"Error clearing feedback: {e}")
            elif choice == "4":
                break
            else:
                print("Invalid choice")

    def _export_feedback(self):
        """
        Export feedback data to a specified directory.
        """
        export_dir = input("Enter export directory path [default: feedback_cache/export]: ").strip()
        if not export_dir:
            export_dir = os.path.join("feedback_cache", "export")

        try:
            self.analyzer.feedback_manager.export_feedback(export_dir)
            self.logger.info(f"Feedback exported to {export_dir}")
            print(f"Feedback exported to {export_dir}")
        except Exception as e:
            self.logger.error(f"Error exporting feedback: {e}")
            print(f"Error exporting feedback: {e}")

    def _import_feedback(self):
        """
        Import feedback data from a specified directory.
        """
        import_dir = input("Enter import directory path: ").strip()
        if not import_dir or not os.path.exists(import_dir):
            self.logger.error("Invalid or non-existent import directory")
            print("Invalid or non-existent directory")
            return

        try:
            self.analyzer.feedback_manager.import_feedback(import_dir)
            self.logger.info(f"Feedback imported from {import_dir}")
            print(f"Feedback imported from {import_dir}")
        except Exception as e:
            self.logger.error(f"Error importing feedback: {e}")
            print(f"Error importing feedback: {e}")