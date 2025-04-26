"""
schema_manager.py

Manages database schema metadata extraction and caching for the TableIdentifier-v2.1 application.
Supports SQL Server, PostgreSQL, and MySQL with database-agnostic queries using information_schema
and fallbacks for SQL Server-specific metadata. Provides methods to build, cache, and retrieve schema
data, including a list of all tables.

Dependencies:
- pyodbc: For database connections.
- json, os, logging: For file handling and logging.
- collections.defaultdict: For nested dictionary structures.
- datetime: For timestamp handling.
"""

import logging
import os
import json
from collections import defaultdict
from typing import Dict, List
from datetime import datetime
import pyodbc

class SchemaManager:
    """
    Manages database schema metadata extraction and caching for multiple database types.

    Supports SQL Server, PostgreSQL, and MySQL with database-agnostic queries using
    information_schema and fallbacks for SQL Server-specific metadata.

    Attributes:
        logger (logging.Logger): Logger for schema operations.
        db_name (str): Name of the database.
        cache_dir (str): Directory for schema cache.
        cache_file (str): Path to schema cache file.
        config_file (str): Path to configuration file.
        db_type (str): Detected database type (sqlserver, postgresql, mysql, or generic).
        excluded_schemas (list): Schemas to exclude (e.g., ['information_schema', 'sys']).
    """

    def __init__(self, db_name: str):
        """
        Initialize with database name and logging.

        Args:
            db_name (str): Name of the database.
        """
        self.logger = logging.getLogger("schema")
        self.db_name = db_name
        self.cache_dir = os.path.join("schema_cache", db_name)
        self.cache_file = os.path.join(self.cache_dir, "schema.json")
        self.config_file = os.path.join("app-config", db_name, "config.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.db_type = None
        self.excluded_schemas = self._load_excluded_schemas()
        self.logger.debug(f"Initialized SchemaManager for {db_name}")
        self.logger.info("SchemaManager version: 2025-04-25 with enhanced database support")

    def _load_excluded_schemas(self) -> list:
        """
        Load excluded schemas from config file or use defaults.

        Returns:
            list: List of schemas to exclude (e.g., ['information_schema', 'sys', 'dbo']).
        """
        default_excluded = ['information_schema', 'sys', 'dbo', 'pg_catalog']
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('excluded_schemas', default_excluded)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        return default_excluded

    def set_db_type(self, connection: pyodbc.Connection):
        """
        Detect database type from connection.

        Args:
            connection: Database connection object.
        """
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0].lower()
            if "microsoft sql server" in version:
                self.db_type = "sqlserver"
            elif "postgresql" in version:
                self.db_type = "postgresql"
            elif "mysql" in version:
                self.db_type = "mysql"
            else:
                self.db_type = "generic"
            self.logger.debug(f"Detected database type: {self.db_type}, version: {version}")
        except Exception as e:
            self.logger.error(f"Error detecting database type: {e}")
            self.db_type = "generic"
        finally:
            cursor.close()

    def needs_refresh(self, connection: pyodbc.Connection) -> bool:
        """
        Check if schema cache needs refreshing based on schema changes.

        Args:
            connection: Database connection object.

        Returns:
            bool: True if refresh is needed, False otherwise.
        """
        try:
            if not self.db_type:
                self.set_db_type(connection)
            schema_mtime = self._get_schema_mtime(connection)
            cache_mtime = os.path.getmtime(self.cache_file) if os.path.exists(self.cache_file) else 0
            self.logger.debug(f"Latest schema change: {schema_mtime}, Cache mtime: {cache_mtime}")
            return schema_mtime > cache_mtime
        except Exception as e:
            self.logger.error(f"Error checking schema refresh: {e}")
            return True

    def _get_schema_mtime(self, connection: pyodbc.Connection) -> float:
        """
        Get the latest schema modification timestamp.

        Uses database-specific queries for SQL Server, PostgreSQL, and MySQL, with a generic fallback.

        Args:
            connection: Database connection object.

        Returns:
            float: Latest modification timestamp or 0 if error occurs or no data.
        """
        cursor = connection.cursor()
        try:
            if self.db_type == "sqlserver":
                query = """
                    SELECT MAX(last_update)
                    FROM (
                        SELECT MAX(create_date) as last_update
                        FROM sys.objects
                        WHERE type = 'U' AND schema_name(schema_id) NOT IN (%s)
                        UNION
                        SELECT MAX(modify_date)
                        FROM sys.objects
                        WHERE type = 'U' AND schema_name(schema_id) NOT IN (%s)
                    ) AS t
                """ % (",".join(f"'{s}'" for s in self.excluded_schemas), ",".join(f"'{s}'" for s in self.excluded_schemas))
                self.logger.debug(f"Executing SQL Server schema mtime query: {query}")
                cursor.execute(query)
                result = cursor.fetchone()
                mtime = result[0].timestamp() if result and result[0] else 0
                self.logger.debug(f"Schema mtime retrieved: {mtime}")
                return mtime
            elif self.db_type == "postgresql":
                query = """
                    SELECT MAX(last_update)
                    FROM (
                        SELECT MAX(create_date) as last_update
                        FROM information_schema.tables
                        WHERE table_schema NOT IN (%s)
                        UNION
                        SELECT MAX(last_altered)
                        FROM information_schema.tables
                        WHERE table_schema NOT IN (%s)
                    ) AS t
                """ % (",".join(f"'{s}'" for s in self.excluded_schemas), ",".join(f"'{s}'" for s in self.excluded_schemas))
                self.logger.debug(f"Executing PostgreSQL schema mtime query: {query}")
                cursor.execute(query)
                result = cursor.fetchone()
                mtime = result[0].timestamp() if result and result[0] else 0
                self.logger.debug(f"Schema mtime retrieved: {mtime}")
                return mtime
            elif self.db_type == "mysql":
                query = """
                    SELECT MAX(UPDATE_TIME)
                    FROM information_schema.tables
                    WHERE TABLE_SCHEMA NOT IN (%s)
                """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
                self.logger.debug(f"Executing MySQL schema mtime query: {query}")
                cursor.execute(query)
                result = cursor.fetchone()
                mtime = result[0].timestamp() if result and result[0] else 0
                self.logger.debug(f"Schema mtime retrieved: {mtime}")
                return mtime
            else:
                self.logger.warning("Generic database type, schema mtime not supported")
                return 0
        except Exception as e:
            self.logger.error(f"Error getting schema mtime: {e}")
            return 0
        finally:
            cursor.close()

    def build_data_dict(self, connection: pyodbc.Connection) -> Dict:
        """
        Build a comprehensive schema dictionary from the database.

        Excludes system schemas and uses information_schema for portability.

        Args:
            connection: Database connection object.

        Returns:
            Dict: Schema dictionary with tables, columns, indexes, foreign keys, and views.
        """
        self.set_db_type(connection)
        cursor = connection.cursor()
        schema_dict = {
            "tables": defaultdict(list),
            "columns": defaultdict(lambda: defaultdict(dict)),
            "indexes": defaultdict(lambda: defaultdict(list)),
            "foreign_keys": defaultdict(lambda: defaultdict(list)),
            "views": defaultdict(list),
            "version": "1.1"  # Updated version for enhanced schema
        }
        
        try:
            # Fetch tables
            query = """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN (%s)
            """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
            cursor.execute(query)
            for schema, table in cursor.fetchall():
                schema_dict["tables"][schema].append(table)
            
            # Fetch columns with constraint information
            query = """
                SELECT c.table_schema, c.table_name, c.column_name, c.data_type,
                       c.is_nullable, c.column_default,
                       CASE
                           WHEN pk.constraint_type = 'PRIMARY KEY' THEN 'PRIMARY KEY'
                           WHEN fk.constraint_type = 'FOREIGN KEY' THEN 'FOREIGN KEY'
                           ELSE NULL
                       END as constraint_type
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT kcu.table_schema, kcu.table_name, kcu.column_name, tc.constraint_type
                    FROM information_schema.key_column_usage kcu
                    JOIN information_schema.table_constraints tc
                    ON kcu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                ) pk ON c.table_schema = pk.table_schema
                AND c.table_name = pk.table_name
                AND c.column_name = pk.column_name
                LEFT JOIN (
                    SELECT kcu.table_schema, kcu.table_name, kcu.column_name, tc.constraint_type
                    FROM information_schema.key_column_usage kcu
                    JOIN information_schema.table_constraints tc
                    ON kcu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                ) fk ON c.table_schema = fk.table_schema
                AND c.table_name = fk.table_name
                AND c.column_name = fk.column_name
                WHERE c.table_schema NOT IN (%s)
            """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
            cursor.execute(query)
            for schema, table, column, dtype, nullable, default, cons_type in cursor.fetchall():
                schema_dict["columns"][schema][table][column] = {
                    "type": dtype,
                    "nullable": nullable == "YES",
                    "default": default,
                    "is_primary_key": cons_type == "PRIMARY KEY",
                    "is_foreign_key": cons_type == "FOREIGN KEY"
                }
            
            # Fetch indexes
            if self.db_type == "sqlserver":
                query = """
                    SELECT schema_name(t.schema_id) AS table_schema,
                           t.name AS table_name,
                           i.name AS index_name,
                           c.name AS column_name
                    FROM sys.indexes i
                    JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
                    JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
                    JOIN sys.tables t ON i.object_id = t.object_id
                    WHERE schema_name(t.schema_id) NOT IN (%s)
                """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
                cursor.execute(query)
            elif self.db_type == "postgresql":
                query = """
                    SELECT n.nspname AS table_schema,
                           t.relname AS table_name,
                           i.relname AS index_name,
                           a.attname AS column_name
                    FROM pg_index ix
                    JOIN pg_class i ON i.oid = ix.indexrelid
                    JOIN pg_class t ON t.oid = ix.indrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                    WHERE n.nspname NOT IN (%s)
                """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
                cursor.execute(query)
            elif self.db_type == "mysql":
                query = """
                    SELECT table_schema, table_name, index_name, column_name
                    FROM information_schema.statistics
                    WHERE table_schema NOT IN (%s)
                """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
                cursor.execute(query)
            else:
                query = """
                    SELECT table_schema, table_name, index_name, column_name
                    FROM information_schema.statistics
                    WHERE table_schema NOT IN (%s)
                """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
                cursor.execute(query)
            for schema, table, index_name, column in cursor.fetchall():
                schema_dict["indexes"][schema][table].append({
                    "index_name": index_name,
                    "column": column
                })
            
            # Fetch foreign keys
            query = """
                SELECT tc.table_schema, tc.table_name, kcu.column_name,
                       ccu.table_schema AS ref_schema, ccu.table_name AS ref_table,
                       ccu.column_name AS ref_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema NOT IN (%s)
            """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
            cursor.execute(query)
            for schema, table, column, ref_schema, ref_table, ref_column in cursor.fetchall():
                schema_dict["foreign_keys"][schema][table].append({
                    "column": column,
                    "referenced_table": f"{ref_schema}.{ref_table}",
                    "referenced_column": ref_column
                })
            
            # Fetch views
            query = """
                SELECT table_schema, table_name
                FROM information_schema.views
                WHERE table_schema NOT IN (%s)
            """ % ",".join(f"'{s}'" for s in self.excluded_schemas)
            cursor.execute(query)
            for schema, view in cursor.fetchall():
                schema_dict["views"][schema].append(view)
            
            self._validate_schema(schema_dict)
            
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            self.logger.debug(f"Saved schema to {self.cache_file}")
            
            return schema_dict
        except Exception as e:
            self.logger.error(f"Error building schema: {e}")
            raise
        finally:
            cursor.close()

    def _validate_schema(self, schema_dict: Dict):
        """
        Validate schema consistency to ensure metadata integrity.

        Args:
            schema_dict: Schema dictionary to validate.
        """
        for schema, tables in schema_dict["tables"].items():
            for table in tables:
                if table not in schema_dict["columns"][schema]:
                    self.logger.warning(f"Table {schema}.{table} has no columns defined")
                for fk in schema_dict["foreign_keys"][schema][table]:
                    ref_table = fk["referenced_table"]
                    ref_schema, ref_table_name = ref_table.split('.')
                    if ref_table_name not in schema_dict["tables"].get(ref_schema, []):
                        self.logger.warning(f"Foreign key references non-existent table {ref_table}")
        self.logger.debug("Schema validation completed")

    def load_from_cache(self) -> Dict:
        """
        Load schema from cache file if available and valid.

        Returns:
            Dict: Cached schema dictionary or empty schema if cache is invalid.
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    schema_dict = json.load(f)
                    self._validate_schema(schema_dict)
                    self.logger.debug(f"Loaded schema from {self.cache_file}")
                    return schema_dict
        except Exception as e:
            self.logger.error(f"Error loading schema from cache: {e}")
        return {
            "tables": defaultdict(list),
            "columns": defaultdict(lambda: defaultdict(dict)),
            "indexes": defaultdict(lambda: defaultdict(list)),
            "foreign_keys": defaultdict(lambda: defaultdict(list)),
            "views": defaultdict(list),
            "version": "1.1"
        }

    def get_all_tables(self) -> List[str]:
        """
        Get a list of all table names in the schema.

        Returns:
            List[str]: List of table names in the format 'schema.table'.
        """
        schema_dict = self.load_from_cache()
        tables = []
        for schema, table_list in schema_dict["tables"].items():
            for table in table_list:
                tables.append(f"{schema}.{table}")
        self.logger.debug(f"Retrieved {len(tables)} tables: {tables}")
        return tables