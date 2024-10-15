import logging
from logging.config import dictConfig

_logconfig_dict_default = {
    "version": 1,
    "incremental": False,
    "disable_existing_loggers": False,
    "root": {
        "level": "INFO",
        "handlers": ["timed_rotating_file_handler", "console"]
    },
    "formatters": {
        "default": {
            "class": "logging.Formatter",
            "format": "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S %z"
        }
    },
    "handlers": {
        "timed_rotating_file_handler": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": "logs/log.log",
            "when": "D",
            "backupCount": 7,
            "interval": 1,
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
    },
    "loggers": {
        "generic": {
            "level": "INFO",
            "handlers": ["timed_rotating_file_handler", "console"],
            "propagate": False,
            "qualname": "generic"
        }
    }
}

dictConfig(_logconfig_dict_default)
logger = logging.getLogger()