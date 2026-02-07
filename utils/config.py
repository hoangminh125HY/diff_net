"""
Configuration Loader - Load and manage model configurations from YAML files
"""
import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration class for loading and managing YAML configs"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        print(f"Loaded configuration from: {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots)
        
        Args:
            key: Configuration key (e.g., 'model.name' or 'enhancement.embed_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key (supports nested keys with dots)
        
        Args:
            key: Configuration key (e.g., 'model.name')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: str = None):
        """
        Save configuration to YAML file
        
        Args:
            save_path: Path to save configuration (if None, uses original path)
        """
        if save_path is None:
            save_path = self.config_path
        
        if save_path is None:
            raise ValueError("No save path specified and no original path available")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {save_path}")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary
        
        Args:
            updates: Dictionary of updates
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def print_config(self):
        """Print the current configuration"""
        print("Current Configuration:")
        print("=" * 50)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            'enhancement': self.get('enhancement', {}),
            'depth': self.get('depth', {}),
            'evolution': self.get('evolution', {}),
            'detection': self.get('detection', {})
        }


def load_config(config_name: str = 'base') -> Config:
    """
    Load configuration by name
    
    Args:
        config_name: Configuration name ('tiny', 'small', 'base', 'large')
        
    Returns:
        Config instance
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to project root and into configs
    config_dir = os.path.join(os.path.dirname(current_dir), 'configs')
    
    config_path = os.path.join(config_dir, f'model_{config_name}.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return Config(config_path)


def get_available_configs() -> list:
    """
    Get list of available configuration files
    
    Returns:
        List of configuration names
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(os.path.dirname(current_dir), 'configs')
    
    if not os.path.exists(config_dir):
        return []
    
    config_files = [f for f in os.listdir(config_dir) if f.startswith('model_') and f.endswith('.yaml')]
    config_names = [f.replace('model_', '').replace('.yaml', '') for f in config_files]
    
    return sorted(config_names)


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Configuration Loader")
    print("=" * 50)
    
    # List available configs
    available_configs = get_available_configs()
    print(f"Available configurations: {available_configs}")
    
    # Load and test each config
    for config_name in available_configs:
        print(f"\n--- Testing {config_name.upper()} config ---")
        config = load_config(config_name)
        
        print(f"Model name: {config.get('model.name')}")
        print(f"Enhancement embed_dim: {config.get('enhancement.embed_dim')}")
        print(f"Depth embed_dim: {config.get('depth.embed_dim')}")
        print(f"Number of classes: {config.get('model.num_classes')}")
    
    print("\n" + "=" * 50)
    print("Configuration loader test completed!")
