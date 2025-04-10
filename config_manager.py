import json
import yaml
import os

class ConfigManager:
    def __init__(self, config_path=None):
        """
        初始化配置管理器
        :param config_path: 配置文件路径，如果为None则自动检测
        """
        self.config_path = config_path or self._detect_config_file()
        self.config = self._load_config()
    
    def _detect_config_file(self):
        """自动检测配置文件路径"""
        possible_files = [
            'config/config.json',
            'config/config.yaml',
            'config/db.json',
            'config/db.yaml'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError("No configuration file found")
    
    def _load_config(self):
        """根据文件扩展名加载不同格式的配置文件"""
        _, ext = os.path.splitext(self.config_path)
        
        with open(self.config_path, 'r') as f:
            if ext == '.json':
                return json.load(f)
            elif ext in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
    
    def get(self, key, default=None):
        """
        获取配置值
        :param key: 配置键，支持点分格式如'database.host'
        :param default: 默认值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all_config(self):
        """获取全部配置"""
        return self.config
    
    # 兼容旧方法
    def get_host(self):
        return self.get('host')
    
    def get_username(self):
        return self.get('username')
    
    def get_password(self):
        return self.get('password')

cm = ConfigManager()
