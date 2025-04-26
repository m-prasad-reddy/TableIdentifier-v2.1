import pyodbc
from typing import Dict, Optional
import logging
import logging.config
import os

class DatabaseConnection:
    """Manages database connections using pyodbc for multiple database types.

    Supports SQL Server, PostgreSQL, and MySQL connections.
    """

    def __init__(self):
        """Initialize the connection manager."""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("connection")
        self.connection = None
        self.current_config = None
        self.logger.debug("Initialized DatabaseConnection")

    def connect(self, config: Dict) -> bool:
        """Connect to a database using the provided configuration.

        Args:
            config (Dict): Configuration dictionary with server, database, username, password, and driver.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            db_type = config.get('driver', '').lower()
            if 'sql server' in db_type:
                conn_str = (
                    f"DRIVER={{{config['driver']}}};"
                    f"SERVER={config['server']};"
                    f"DATABASE={config['database']};"
                    f"UID={config['username']};"
                    f"PWD={config['password']}"
                )
            elif 'postgresql' in db_type:
                conn_str = (
                    f"DRIVER={{{config['driver']}}};"
                    f"SERVER={config['server']};"
                    f"PORT={config.get('port', '5432')};"
                    f"DATABASE={config['database']};"
                    f"UID={config['username']};"
                    f"PWD={config['password']}"
                )
            elif 'mysql' in db_type:
                conn_str = (
                    f"DRIVER={{{config['driver']}}};"
                    f"SERVER={config['server']};"
                    f"PORT={config.get('port', '3306')};"
                    f"DATABASE={config['database']};"
                    f"UID={config['username']};"
                    f"PWD={config['password']}"
                )
            else:
                raise ValueError(f"Unsupported driver: {config['driver']}")

            self.connection = pyodbc.connect(conn_str)
            self.current_config = config
            self.logger.info(f"Connected to database: {config['database']} with driver {config['driver']}")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            print(f"Connection failed: {str(e)}")
            return False

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.logger.info("Closing database connection")
            self.connection.close()
            self.connection = None
            self.current_config = None

    def is_connected(self) -> bool:
        """Check if the connection is active.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connection is not None

    def get_cursor(self) -> Optional[pyodbc.Cursor]:
        """Get a cursor for the database connection.

        Returns:
            Optional[pyodbc.Cursor]: Cursor object if connected, None otherwise.
        """
        return self.connection.cursor() if self.connection else None