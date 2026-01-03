"""
Query orchestrator for ST-Bot - Simplified MVP.

Coordinates the flow: intent classification → handler routing → response building.
Simplified to 5 core intents and 3 handlers.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.context import ConversationContext, Intent, IntentType, Product
from core.intent import IntentClassifier
from core.filters import FilterExtractor
from core.structured_logging import log_conversation_turn
from core.conversation_csv import log_conversation as log_conversation_csv
from core.gsheets_logger import log_to_gsheets
from ui.state import SessionState
from ui.responses import ResponseFormatter
from ui.logging import ConversationLogger
from llm.product_ranker import ProductRanker
from llm.response_builder import ResponseBuilder
from llm.query_analyzer import QueryAnalyzer

from handlers.base import HandlerContext, HandlerResult
from handlers.greeting import GreetingHandler, FarewellHandler
from handlers.search import NewSearchHandler
from handlers.followup import FollowupHandler


@dataclass
class OrchestratorComponents:
    """
    All components needed by the orchestrator - Simplified MVP.

    These are typically created by the main app and passed to the orchestrator.
    The search_engine should be a SearchEngineWrapper or similar that provides
    a .search(filters) interface.
    """
    intent_classifier: Any  # IntentClassifier
    filter_extractor: Any   # FilterExtractor
    search_engine: Any      # SearchEngineWrapper (has .search(filters) method)
    product_ranker: Any     # ProductRanker
    response_builder: Any   # ResponseBuilder
    formatter: Any          # ResponseFormatter
    query_analyzer: Any     # QueryAnalyzer
    logger: Any             # ConversationLogger


def create_components_from_tuple(components_tuple) -> OrchestratorComponents:
    """
    Create OrchestratorComponents from the existing get_components() tuple.

    This bridges the existing component creation pattern to the new structure.

    Args:
        components_tuple: Tuple from get_components() function

    Returns:
        OrchestratorComponents instance
    """
    # Handle both old format (10 items) and new simplified format (8 items)
    if len(components_tuple) == 10:
        # Old format with domain_rules and device_inference
        (classifier, filter_extractor, search_engine, domain_rules,
         formatter, logger, device_inference, product_ranker,
         response_builder, query_analyzer) = components_tuple
    else:
        # New simplified format
        (classifier, filter_extractor, search_engine,
         formatter, logger, product_ranker,
         response_builder, query_analyzer) = components_tuple

    return OrchestratorComponents(
        intent_classifier=classifier,
        filter_extractor=filter_extractor,
        search_engine=search_engine,
        product_ranker=product_ranker,
        response_builder=response_builder,
        formatter=formatter,
        query_analyzer=query_analyzer,
        logger=logger,
    )


# Handler registry - maps intent types to handlers (simplified to 4 handlers)
HANDLERS = {
    IntentType.GREETING: GreetingHandler(),
    IntentType.FAREWELL: FarewellHandler(),
    IntentType.NEW_SEARCH: NewSearchHandler(),
    IntentType.FOLLOWUP: FollowupHandler(),
    IntentType.AMBIGUOUS: FollowupHandler(),  # Treat ambiguous as followup
}


def process_query(
    query: str,
    context: ConversationContext,
    session: SessionState,
    components: OrchestratorComponents,
    all_products: List[Product],
    st_session_state: Any,
    debug_mode: bool = False
) -> Tuple[str, str]:
    """
    Process a user query and return a response.

    This is the main entry point - simplified to 3 core flows:
    1. Greetings/Farewells → Simple responses
    2. New Search → Find products
    3. Followup → Refine/answer questions about products

    Args:
        query: User's query text
        context: Conversation context (products shown, etc.)
        session: Session state (message history)
        components: All orchestrator components
        all_products: List of all products for searching
        st_session_state: Streamlit's session state for persistence
        debug_mode: Whether to include debug output

    Returns:
        Tuple of (response_text, intent_type_value)
    """
    # Start timing for response
    start_time = time.perf_counter()

    debug_lines = []

    # Debug: Show context state
    if debug_mode:
        context_count = len(context.current_products) if context.current_products else 0
        debug_lines.append(f"📦 CONTEXT: {context_count} products in context")
        if context.current_products:
            sku_list = [p.product_number for p in context.current_products[:5]]
            debug_lines.append(f"   SKUs: {', '.join(sku_list)}")

    # Step 1: Classify intent
    intent = components.intent_classifier.classify(query, context)
    if debug_mode:
        debug_lines.append(f"🎯 INTENT: {intent.type.value} (confidence={intent.confidence:.2f})")

    # Step 2: Get handler for intent
    handler = HANDLERS.get(intent.type)
    if not handler:
        # Fallback to new search for unknown intents
        handler = NewSearchHandler()
        if debug_mode:
            debug_lines.append(f"⚠️ No handler for {intent.type.value}, using NewSearchHandler")

    # Step 3: Build handler context
    handler_ctx = HandlerContext(
        query=query,
        intent=intent,
        context=context,
        session=session,
        all_products=all_products,
        st_session_state=st_session_state,
        debug_mode=debug_mode,
        filter_extractor=components.filter_extractor,
        search_engine=components.search_engine,
        product_ranker=components.product_ranker,
        response_builder=components.response_builder,
        formatter=components.formatter,
        query_analyzer=components.query_analyzer,
        debug_lines=debug_lines,
    )

    # Step 4: Execute handler
    try:
        result = handler.handle(handler_ctx)
    except Exception as e:
        # Log error and return graceful response
        if debug_mode:
            debug_lines.append(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        result = HandlerResult(
            response="I encountered an issue processing your request. "
                     "Could you try rephrasing your question?"
        )

    # Step 5: Apply side effects from result
    if result.products_to_set:
        context.set_multi_products(result.products_to_set)
        session.set_product_context(result.products_to_set, intent.type)
        if debug_mode:
            debug_lines.append(f"💾 SAVED: {len(result.products_to_set)} products to context")

    # Step 6: Log conversation (both structured and CSV logging)
    # For logging, use products_to_set if handler set new products,
    # otherwise fall back to context.current_products (for followups that discuss existing products)
    products_for_logging = result.products_to_set if result.products_to_set else context.current_products
    products_shown_count = len(products_for_logging) if products_for_logging else 0
    product_skus = [p.product_number for p in products_for_logging] if products_for_logging else []
    response_time_ms = (time.perf_counter() - start_time) * 1000

    # Log to CSV legacy logger
    components.logger.log_conversation(
        session_id=session.session_id,
        user_message=query,
        bot_response=result.response,
        intent_type=intent.type.value,
        products_shown=products_shown_count
    )

    # Log comprehensive conversation turn for Power BI (to structured log)
    log_conversation_turn(
        session_id=session.session_id,
        user_query=query,
        intent_result=intent.type.value,
        intent_confidence=intent.confidence,
        products_found=result.products_found,
        products_shown=products_shown_count,
        product_skus=product_skus,
        filters=result.filters_for_logging,
        response_time_ms=response_time_ms
    )

    # Log to clean conversations.csv (one row per user interaction)
    # Use result.response (clean response without debug header)
    log_conversation_csv(
        session_id=session.session_id,
        user_query=query,
        bot_response=result.response,
        intent=intent.type.value,
        confidence=intent.confidence,
        filters=result.filters_for_logging,
        products_found=result.products_found,
        products_shown=products_shown_count,
        product_skus=product_skus,
        response_time_ms=response_time_ms
    )

    # Log to Google Sheets (for cloud deployment)
    # This is a no-op if Google Sheets is not configured
    log_to_gsheets(
        session_id=session.session_id,
        user_query=query,
        bot_response=result.response,
        intent=intent.type.value,
        confidence=intent.confidence,
        filters=result.filters_for_logging,
        products_found=result.products_found,
        products_shown=products_shown_count,
        product_skus=product_skus,
        response_time_ms=response_time_ms
    )

    # Step 7: Build final response
    response = result.response

    # Add debug output if enabled
    if debug_mode and debug_lines:
        # debug_lines is shared with handler_ctx, so all debug is in one list
        debug_header = "**🔍 DEBUG OUTPUT:**\n```\n" + "\n".join(debug_lines) + "\n```\n\n---\n\n"
        response = debug_header + response

    return response, intent.type.value


class QueryOrchestrator:
    """
    Class-based orchestrator for dependency injection.

    Use this when you need to customize components or for testing.
    """

    def __init__(
        self,
        components: OrchestratorComponents,
        all_products: List[Product],
        debug_mode: bool = False
    ):
        self.components = components
        self.all_products = all_products
        self.debug_mode = debug_mode

    def process(
        self,
        query: str,
        context: ConversationContext,
        session: SessionState,
        st_session_state: Any
    ) -> Tuple[str, str]:
        """
        Process a query using this orchestrator's components.

        Args:
            query: User's query text
            context: Conversation context
            session: Session state
            st_session_state: Streamlit's session state

        Returns:
            Tuple of (response_text, intent_type_value)
        """
        return process_query(
            query=query,
            context=context,
            session=session,
            components=self.components,
            all_products=self.all_products,
            st_session_state=st_session_state,
            debug_mode=self.debug_mode
        )
