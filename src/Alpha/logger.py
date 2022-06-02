import os
import logging

FILE = "game.log"
IS_KAGGLE = os.path.exists("/kaggle_simulations")
LEVEL = logging.DEBUG if not IS_KAGGLE else logging.INFO
LOGGING_ENABLED = True


class _FileHandler(logging.FileHandler):
    def emit(self, record):
        if not LOGGING_ENABLED:
            return

        print(self.format(record))
        if not IS_KAGGLE:
            super().emit(record)


def init_logger(_logger):
    if not LOGGING_ENABLED:
        return

    if not IS_KAGGLE:
        if os.path.exists(FILE):
            os.remove(FILE)

    while _logger.hasHandlers():
        _logger.removeHandler(_logger.handlers[0])

    _logger.setLevel(LEVEL)
    ch = _FileHandler(FILE)
    ch.setLevel(LEVEL)
    formatter = logging.Formatter(
        "%(levelname)s - %(message)s", datefmt="%H-%M-%S"
    )
    ch.setFormatter(formatter)
    _logger.addHandler(ch)


if IS_KAGGLE:
    logger = logging.getLogger()
else:
    logger = logging.getLogger("Alpha")
