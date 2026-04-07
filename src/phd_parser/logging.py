import logging

def setup_logger(use_file_handler=True, use_console_handler=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers first
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
        elif isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # Now add the requested handlers
    if use_file_handler:
        file_handler = logging.FileHandler('phd_parser.log', mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s'))
        logger.addHandler(file_handler)
    
    if use_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(name)s %(message)s'))
        logger.addHandler(console_handler)