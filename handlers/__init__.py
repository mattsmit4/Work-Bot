"""
Intent handlers for ST-Bot - Simplified MVP.

Each handler processes a specific type of user intent.
"""

from handlers.base import BaseHandler, HandlerContext, HandlerResult
from handlers.greeting import GreetingHandler, FarewellHandler
from handlers.search import NewSearchHandler
from handlers.followup import FollowupHandler

__all__ = [
    # Base classes
    'BaseHandler',
    'HandlerContext',
    'HandlerResult',
    # Handlers
    'GreetingHandler',
    'FarewellHandler',
    'NewSearchHandler',
    'FollowupHandler',
]
