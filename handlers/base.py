"""
Base handler and context classes for ST-Bot intent handlers.

Provides the common interface and shared context for all handlers.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Callable
from abc import ABC, abstractmethod

from core.context import (
    ConversationContext, Intent, Product, SearchFilters, SearchResult
)
from ui.state import SessionState


@dataclass
class HandlerContext:
    """
    Context passed to all intent handlers.

    Contains everything a handler needs to process a query:
    - The query itself
    - Classified intent
    - Conversation context (products shown, pending guidance, etc.)
    - Session state (message history)
    - All products for searching
    - Streamlit session state for persistence
    - Component references

    This avoids passing dozens of parameters to each handler.
    """
    query: str
    intent: Intent
    context: ConversationContext
    session: SessionState
    all_products: List[Product]
    st_session_state: Any  # Streamlit's st.session_state
    debug_mode: bool = False

    # Component references (set by orchestrator)
    filter_extractor: Any = None
    search_engine: Any = None
    product_ranker: Any = None
    response_builder: Any = None
    formatter: Any = None
    query_analyzer: Any = None

    # Debug output collector
    debug_lines: List[str] = field(default_factory=list)

    def add_debug(self, message: str) -> None:
        """Add a debug message."""
        if self.debug_mode:
            self.debug_lines.append(message)


@dataclass
class HandlerResult:
    """
    Result returned by intent handlers.

    Contains the response text and any side effects:
    - Products to set in context
    - Whether to save guidance state
    - Whether to save pending question state
    """
    response: str
    products_to_set: Optional[List[Product]] = None
    save_guidance: bool = False
    save_pending_question: bool = False
    clear_guidance: bool = False
    clear_pending_question: bool = False


class BaseHandler(ABC):
    """
    Base class for all intent handlers.

    Each handler processes a specific intent type and returns a HandlerResult.
    Handlers should be stateless - all state is in HandlerContext.
    """

    @abstractmethod
    def handle(self, ctx: HandlerContext) -> HandlerResult:
        """
        Process the intent and return a result.

        Args:
            ctx: Handler context with query, intent, and all components

        Returns:
            HandlerResult with response and any side effects
        """
        pass

    def _clear_stale_context(self, ctx: HandlerContext) -> None:
        """
        Clear stale context when starting a new flow.

        Call this when handling intents that start fresh (new search,
        explicit SKU, new guidance flow).
        """
        # Clear pending feature search offers
        if ctx.context.has_pending_feature_search():
            ctx.add_debug(f"🧹 CLEARING STALE FEATURE SEARCH: {ctx.context.pending_feature_search}")
            ctx.context.clear_pending_feature_search()

        # Clear comparison context
        if ctx.context.has_comparison_context():
            ctx.add_debug(f"🧹 CLEARING STALE COMPARISON: {ctx.context.last_comparison_indices}")
            ctx.context.clear_comparison_context()

        # Clear pending questions
        if ctx.context.has_pending_question():
            ctx.add_debug(f"🧹 CLEARING STALE PENDING QUESTION: {ctx.context.pending_question.question_type}")
            ctx.context.clear_pending_question()
