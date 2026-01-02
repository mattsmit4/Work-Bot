"""
Greeting and farewell intent handlers.

Simple handlers for conversational intents that don't require
product search or complex logic.
"""

from handlers.base import BaseHandler, HandlerContext, HandlerResult


class GreetingHandler(BaseHandler):
    """Handle greeting intent."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        response = ctx.formatter.format_greeting()
        return HandlerResult(response=response)


class FarewellHandler(BaseHandler):
    """Handle farewell intent."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        response = ctx.formatter.format_farewell()
        return HandlerResult(response=response)
