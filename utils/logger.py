import logging
import os
from datetime import datetime

class Logger:
    """Logging utility for HSGSP"""
    
    def __init__(self, config):
        self.config = config
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logging configuration"""
        logs_dir = os.path.join(self.config.logs_dir,
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, "HSGSP.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('HSGSP')
    
    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)