"""The package logger keeps writing after a library has closed every handler."""
import logging

import phd_parser as pp


def test_file_handler_survives_logging_shutdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pp.logging.setup_logger(level="DEBUG", use_file_handler=True, use_console_handler=False)
    logger = logging.getLogger("phd_parser.test")
    logger.info("before")
    logging.shutdown()          # what SpectroChemPy does on its first read
    logger.info("after")
    for handler in logging.getLogger("phd_parser").handlers:
        handler.flush()
    text = (tmp_path / "phd_parser.log").read_text(encoding="utf-8")
    assert "before" in text and "after" in text
    pp.logging.setup_logger(use_file_handler=False)
