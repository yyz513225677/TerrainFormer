"""Logging utilities"""

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: str = 'outputs', name: str = 'terrainformer'):
    """Setup logging configuration."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/{name}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(name)
