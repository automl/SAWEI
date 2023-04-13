import logging

from rich import print


def configure_smac_loggers(level: int = logging.ERROR, detach: bool = False):
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers = [l for l in loggers if l.name.startswith("smac.")]

    for logger in loggers:
        logger.setLevel(level=level)
        logger.propagate = False

        if detach:
            _handlers = logger.handlers
            for _handler in _handlers:
                logger.removeHandler(_handler)
