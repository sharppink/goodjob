"""
config package
Exposes the singleton settings object for use throughout the application.
"""

from config.settings import Settings

settings = Settings()

__all__ = ["settings"]
