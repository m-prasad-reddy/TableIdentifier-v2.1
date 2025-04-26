import os
import pandas as pd
import json
import logging
from typing import Dict

class Trainer:
    """Manages training data from CSV for table identification.

    Generates and updates training data dynamically based on schema metadata.
    """

    def __init__(self, db_name: str, schema_dict: Dict):
        """Initialize with database name and schema.

        Args:
            db_name (str): Name of the database.
            schema_dict (Dict): Schema dictionary.
        """
        self.logger = logging.getLogger("trainer")
        self.db_name = db_name
        self.schema_dict = schema_dict
        self.trainer_path = os.path.join("app-config", db_name, "db_config_trainer.csv")
        self.training_data = None
        self.load_training_data()
        self.logger.debug(f"Initialized Trainer for {db_name}")

    def load_training_data(self):
        """Load or create training data from CSV."""
        if os.path.exists(self.trainer_path):
            try:
                self.training_data = pd.read_csv(self.trainer_path)
                self.logger.debug(f"Loaded training data from {self.trainer_path}")
            except Exception as e:
                self.logger.error(f"Error loading training data: {e}")
                self._create_template()
        else:
            self._create_template()

    def _create_template(self):
        """Create a template CSV if none exists."""
        columns = [
            "DB_Config", "Schema", "Table_Name", "Primary_Keys", "Foreign_Keys",
            "Associated_Tables", "Associated_Views", "Description", "Columns_List"
        ]
        template_data = []
        for schema in self.schema_dict["tables"]:
            for table in self.schema_dict["tables"][schema]:
                columns_list = ",".join(self.schema_dict["columns"][schema][table].keys())
                foreign_keys = ",".join(
                    f"{fk['column']}->{fk['referenced_table']}.{fk['referenced_column']}"
                    for fk in self.schema_dict["foreign_keys"][schema][table]
                )
                primary_keys = ",".join(
                    col for col, info in self.schema_dict["columns"][schema][table].items()
                    if info["is_primary_key"]
                )
                template_data.append({
                    "DB_Config": self.db_name,
                    "Schema": schema,
                    "Table_Name": table,
                    "Primary_Keys": primary_keys,
                    "Foreign_Keys": foreign_keys,
                    "Associated_Tables": "",
                    "Associated_Views": "",
                    "Description": f"Information about {table.lower()}",
                    "Columns_List": columns_list
                })
        
        df = pd.DataFrame(template_data, columns=columns)
        os.makedirs(os.path.dirname(self.trainer_path), exist_ok=True)
        df.to_csv(self.trainer_path, index=False)
        self.training_data = df
        self.logger.debug(f"Created template CSV at {self.trainer_path}")

    def update_configs(self, pattern_manager, name_match_manager, feedback_manager):
        """Update configs based on training data.

        Args:
            pattern_manager: PatternManager instance.
            name_match_manager: NameMatchManager instance.
            feedback_manager: FeedbackManager instance.
        """
        if self.training_data is None:
            self.logger.warning("No training data loaded")
            return

        # Update name matches
        for _, row in self.training_data.iterrows():
            columns = row["Columns_List"].split(",")
            table = f"{row['Schema']}.{row['Table_Name']}"
            for col in columns:
                if col:
                    name_match_manager.update_synonyms(col.lower(), [table])

        # Update patterns
        for _, row in self.training_data.iterrows():
            desc = row["Description"].lower()
            table = f"{row['Schema']}.{row['Table_Name']}"
            pattern_manager.update_patterns(desc, [table])

        # Update feedback
        for _, row in self.training_data.iterrows():
            desc = row["Description"]
            table = f"{row['Schema']}.{row['Table_Name']}"
            feedback_manager.store_feedback(desc, [table], self.schema_dict)
        self.logger.debug("Updated configs from training data")