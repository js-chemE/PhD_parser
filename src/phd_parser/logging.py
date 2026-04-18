import logging

LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def setup_logger(level: int | str = logging.INFO, use_file_handler=True, use_console_handler=False):

    if isinstance(level, str):
        level = LEVELS.get(level.upper(), logging.INFO)

    pkg_logger = logging.getLogger('phd_parser')
    pkg_logger.setLevel(level)
    pkg_logger.handlers.clear()
    pkg_logger.propagate = False  # Don't bubble up to root logger

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s')

    if use_file_handler:
        fh = logging.FileHandler('phd_parser.log', mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        pkg_logger.addHandler(fh)

    if use_console_handler:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        pkg_logger.addHandler(ch)

def disable_logging():
    logging.getLogger('phd_parser').setLevel(logging.CRITICAL + 1)