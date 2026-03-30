from pathlib import Path
import spectrochempy as spcp

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def unpack_srsx(srsx_path: Path | str, output_dir: Path | str) -> None:
    if isinstance(srsx_path, str):
        srsx_path = Path(srsx_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    import zipfile

    logger.info(f"Unpacking SRSX file: {srsx_path} to {output_dir}")
    with zipfile.ZipFile(srsx_path, "r") as z:
        for info in z.infolist():
            if info.flag_bits & 0x1:
                logger.debug(f"Skipping encrypted file: {info.filename}")
                continue
            z.extract(info, output_dir)
            logger.debug(f"Extracted file: {info.filename} to {output_dir / info.filename}")

    logger.info("Unpacking completed.")

def read_spa(spa_path: Path | str) -> spcp.NDDataset:
    if isinstance(spa_path, str):
        spa_path = Path(spa_path)
    logger.debug(f"Opening SPA files from: {spa_path}")
    spas = spcp.read_omnic(spa_path)
    logger.info(f"Opened {len(spas)} SPA files.")
    return spas