from pathlib import Path
import os
import contextlib

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        #import spectrochempy as spcp
        pass



# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
def read_spa(spa_path: Path | str):
    raise NotImplementedError("Reading SPA files is currently not implemented. This function is a placeholder for future implementation.")
# def read_spa(spa_path: Path | str) -> spcp.NDDataset:
#     spa_path = Path(spa_path)

#     logger.debug(f"Opening SPA files from: {spa_path}")
#     spas = spcp.read_omnic(spa_path)
#     logger.info(f"Opened {len(spas)} SPA files.")
    
#     return spas