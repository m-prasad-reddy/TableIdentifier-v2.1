import os
import json
from typing import Dict
import logging
import logging.config

class DBConfigManager:
    """Manages database configuration loading and validation.

    Loads configurations from a JSON file and ensures they contain required fields for multiple database types.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("config")
        self.logger.debug("Initialized DBConfigManager")

    def load_configs(self, config_path: str) -> Dict:
        """Load database configurations from a JSON file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict: Dictionary of database configurations.

        Raises:
            FileNotFoundError: If the config file is not found.
            ValueError: If the config file is invalid.
        """
        if not os.path.exists(config_path):
            self.logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path) as f:
            configs = json.load(f)
        
        self._validate_configs(configs)
        self.logger.debug(f"Loaded configurations from {config_path}")
        return configs

    def _validate_configs(self, configs: Dict):
        """Validate the structure and content of configurations.

        Args:
            configs (Dict): Dictionary of configurations.

        Raises:
            ValueError: If the configurations are invalid.
        """
        if not isinstance(configs, dict):
            self.logger.error("Config file should contain a dictionary of configurations")
            raise ValueError("Config file should contain a dictionary of configurations")
        
        required_keys = {'server', 'database', 'username', 'password', 'driver'}
        for key, config in configs.items():
            if not isinstance(config, dict):
                self.logger.error(f"Configuration for {key} must be a dictionary")
                raise ValueError(f"Configuration for {key} must be a dictionary")
            if not required_keys.issubset(config.keys()):
                missing = required_keys - set(config.keys())
                self.logger.error(f"Missing keys in {key} config: {', '.join(missing)}")
                raise ValueError(f"Missing keys in {key} config: {', '.join(missing)}")
            if 'postgresql' in config['driver'].lower() or 'mysql' in config['driver'].lower():
                if 'port' not in config:
                    self.logger.warning(f"Port not specified for {key}, using default")