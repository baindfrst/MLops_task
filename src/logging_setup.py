import logging
import sys
from pathlib import Path

def setup_logging(
    level: str = "INFO",
    logfile: str | None = None,
    overwrite: bool = True,
) -> None:
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt))
    handlers.append(stream_handler)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            logfile, mode="w" if overwrite else "a", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(file_handler)

    logging.basicConfig(level=getattr(logging, level.upper()), handlers=handlers)

    for noisy in ["urllib3", "transformers", "datasets"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
