import sys
from loguru import logger

def setup_logger(level: str = "INFO") -> None:
    """Configure loguru logger for the application."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(
        "nexusiq.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )

# Call once at import time with default level; CLI can reconfigure
setup_logger()
