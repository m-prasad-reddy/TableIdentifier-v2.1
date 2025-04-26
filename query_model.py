import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json
import logging
import logging.config

class TableIdentifierClient:
    """Client to query a pre-trained table identifier model using DistilBERT.

    Operates offline without requiring a live database connection.
    """

    def __init__(self, model_path: str, schema_path: str):
        """Initialize with model and schema paths.

        Args:
            model_path (str): Path to the trained DistilBERT model.
            schema_path (str): Path to the schema JSON file.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("query_model")
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            with open(schema_path, 'r') as f:
                self.schema_dict = json.load(f)
                if self.schema_dict.get('version') != "1.1":
                    self.logger.warning(f"Schema version {self.schema_dict.get('version')} may be incompatible")
            self.tables = self._get_all_tables()
            self.logger.debug(f"Initialized TableIdentifierClient with model {model_path}")
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise

    def _get_all_tables(self) -> list:
        """Get all tables from schema.

        Returns:
            list: List of table names (schema.table).
        """
        tables = []
        for schema in self.schema_dict['tables']:
            tables.extend(f"{schema}.{table}" for table in self.schema_dict['tables'][schema])
        self.logger.debug(f"Loaded {len(tables)} tables from schema")
        return tables

    def query(self, query: str) -> list:
        """Query the model for table suggestions.

        Args:
            query (str): The natural language query.

        Returns:
            list: List of suggested table names.
        """
        try:
            inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                max_prob, max_idx = torch.max(probs, dim=1)
                if max_prob > 0.7:
                    self.logger.debug(f"Suggested table: {self.tables[max_idx.item()]}, probability: {max_prob.item()}")
                    return [self.tables[max_idx.item()]]
                else:
                    self.logger.debug(f"No tables suggested for query: {query}, max probability: {max_prob.item()}")
                    return []
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return []

if __name__ == "__main__":
    model_path = "app-config/models/table_identifier_model.pth"
    schema_path = "schema_cache/BikeStores/schema.json"
    
    if not os.path.exists(model_path) or not os.path.exists(schema_path):
        print("Model or schema not found!")
    else:
        try:
            client = TableIdentifierClient(model_path, schema_path)
            query = input("Enter query: ")
            tables = client.query(query)
            if tables:
                print(f"Suggested tables: {tables}")
            else:
                print("I am not yet trained to get relevant tables identified for this context")
        except Exception as e:
            print(f"Error: {e}")