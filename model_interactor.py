import json
import spacy
import logging
import logging.config
import os

class TableIdentificationModel:
    """Standalone model for table identification from natural language queries.

    Uses SpaCy for lightweight processing and operates without a live database connection.
    """

    def __init__(self, model_path: str):
        """Initialize with model path.

        Args:
            model_path (str): Path to the trained model file.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("trainer")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            self.nlp = None
        self.weights = {}
        self.schema_dict = {}
        self.dynamic_matches = {}
        self.default_matches = {}
        self.load_model(model_path)
        self.logger.debug(f"Initialized TableIdentificationModel with {model_path}")

    def load_model(self, model_path: str):
        """Load the trained model from a JSON file.

        Args:
            model_path (str): Path to the model file.

        Raises:
            Exception: If loading fails.
        """
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            self.weights = model_data['weights']
            self.schema_dict = model_data.get('schema_dict', {})
            self.dynamic_matches = model_data.get('dynamic_matches', {})
            self.default_matches = model_data.get('default_matches', {})
            if model_data.get('version') != "1.1":
                self.logger.warning(f"Model version {model_data.get('version')} may be incompatible")
            self.logger.debug(f"Loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def identify_tables(self, query: str) -> list[str] | None:
        """Identify tables for a query using the loaded model and SpaCy.

        Args:
            query (str): The natural language query.

        Returns:
            list[str] | None: List of identified tables, or None if none are found.
        """
        self.logger.debug(f"Identifying tables for query: {query}")
        if not self.nlp:
            self.logger.warning("SpaCy model not loaded, cannot identify tables")
            return None

        try:
            doc = self.nlp(query.lower())
            table_scores = {}
            query_tokens = [token.lemma_ for token in doc if not token.is_stop]

            for schema in self.schema_dict.get('tables', {}):
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    score = 0.0
                    
                    # Direct table name match
                    if table.lower() in query.lower():
                        score += 0.5
                    
                    # Column similarity
                    for col in self.schema_dict['columns'][schema][table]:
                        col_doc = self.nlp(col.lower())
                        for token in doc:
                            if token.similarity(col_doc) > 0.7:
                                score += 0.8
                    
                    # Token weights
                    for token in query_tokens:
                        score += self.weights.get(table_full, {}).get(token, 0.0)
                    
                    if score > 0:
                        table_scores[table_full] = score
            
            sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            selected_tables = [table for table, _ in sorted_tables]
            
            if selected_tables and max(table_scores.values(), default=0) > 0.5:
                self.logger.debug(f"Identified tables: {selected_tables}")
                return selected_tables
            else:
                self.logger.debug("No relevant tables identified")
                return None
        
        except Exception as e:
            self.logger.error(f"Error identifying tables: {e}")
            return None

def main():
    """Run the model interactor CLI."""
    model_path = input("Enter model path [default: models/table_identifier_model.json]: ").strip()
    if not model_path:
        model_path = "models/table_identifier_model.json"
    
    try:
        model = TableIdentificationModel(model_path)
        print("\n=== Table Identification Model ===")
        while True:
            query = input("\nEnter query (or 'exit'): ").strip()
            if query.lower() == 'exit':
                break
            tables = model.identify_tables(query)
            if tables:
                print("\nIdentified Tables:")
                for i, table in enumerate(tables, 1):
                    print(f"{i}. {table}")
            else:
                print("I am not yet trained to get relevant tables identified for this context")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()