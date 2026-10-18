import logging

LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

class ReopeningFileHandler(logging.FileHandler):
    """A ``FileHandler`` that survives a ``logging.shutdown()`` by someone else.

    SpectroChemPy closes every logging handler when it first reads a file.
    A closed ``FileHandler`` that was opened in write mode never reopens, so
    everything the parser logs after the first ``.spa`` read is lost. This
    handler reopens its file in append mode instead, keeping what was written.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is None:
            self.mode = "a"
            self._closed = False
            self.stream = self._open()
        super().emit(record)


def setup_logger(level: int | str = logging.INFO, use_file_handler=True, use_console_handler=False):
    """Configure the ``phd_parser`` package logger.

    Clears any existing handlers on the ``'phd_parser'`` logger and
    attaches new ones according to the arguments.  The logger does not
    propagate to the root logger so that library users retain full
    control over their own logging configuration.

    Parameters
    ----------
    level : int or str, optional
        Logging level for the package logger.  Accepts standard
        :mod:`logging` integer constants (e.g. ``logging.DEBUG``) or
        their string equivalents (e.g. ``"DEBUG"``).  String values are
        case-insensitive.  Defaults to ``logging.INFO``.
    use_file_handler : bool, optional
        When ``True`` (default), attach a
        :class:`ReopeningFileHandler` that writes to
        ``phd_parser.log`` in the current working directory (mode
        ``'w'``, reopened in append mode if a library closes it).
    use_console_handler : bool, optional
        When ``True``, attach a :class:`logging.StreamHandler` that
        writes to ``stderr`` (default is ``False``).
    """
    if isinstance(level, str):
        level = LEVELS.get(level.upper(), logging.INFO)

    pkg_logger = logging.getLogger('phd_parser')
    pkg_logger.setLevel(level)
    pkg_logger.handlers.clear()
    pkg_logger.propagate = False  # Don't bubble up to root logger

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s')

    if use_file_handler:
        fh = ReopeningFileHandler('phd_parser.log', mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        pkg_logger.addHandler(fh)

    if use_console_handler:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        pkg_logger.addHandler(ch)

def disable_logging():
    """Silence all ``phd_parser`` log output.

    Sets the package logger level above ``CRITICAL`` so that no
    messages at any standard level are emitted.  Call
    :func:`setup_logger` to re-enable logging.
    """
    logging.getLogger('phd_parser').setLevel(logging.CRITICAL + 1)
