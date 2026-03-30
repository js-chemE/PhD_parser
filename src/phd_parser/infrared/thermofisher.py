from pathlib import Path
import spectrochempy as spcp

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def read_spa(spa_path: Path | str) -> spcp.NDDataset:
    spa_path = Path(spa_path)

    logger.debug(f"Opening SPA files from: {spa_path}")
    spas = spcp.read_omnic(spa_path)
    logger.info(f"Opened {len(spas)} SPA files.")
    
    return spas