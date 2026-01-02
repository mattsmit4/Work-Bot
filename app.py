"""
ST-Bot Streamlit App - StarTech.com Product Assistant
Powered by your 4,178 product Excel catalog

Run with: streamlit run app_startech.py

Architecture:
- This file: Streamlit UI only (~150 lines)
- core/orchestrator.py: Query processing coordination
- handlers/: Intent-specific handlers
- core/: Business logic (intent, filters, search)
- llm/: LLM-powered features (ranking, responses)
- ui/: UI helpers (state, formatting, logging)
"""

import streamlit as st
from excel_loader import load_startech_products, get_product_statistics
from core.context import ConversationContext, LengthPreference
from core.intent import IntentClassifier
from core.filters import FilterExtractor
from core.search import SearchStrategy
from core.structured_logging import setup_logging, get_logger
from core.orchestrator import process_query, OrchestratorComponents
from llm.query_analyzer import QueryAnalyzer
from llm.product_ranker import ProductRanker
from llm.response_builder import ResponseBuilder
from ui.responses import ResponseFormatter
from ui.state import SessionState
from ui.logging import ConversationLogger


# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG_MODE = False  # Set to True for development debugging

# Initialize structured logging
setup_logging(
    log_dir="logs",
    console_level=20,  # INFO
    file_level=10,     # DEBUG
    enable_console=True,
    enable_file=True,
)
app_logger = get_logger("app")

# Streamlit page config
st.set_page_config(
    page_title="ST-Bot - StarTech.com Assistant",
    page_icon="🤖",
    layout="wide"
)


# =============================================================================
# COMPONENT INITIALIZATION
# =============================================================================

@st.cache_resource
def load_products(excel_path: str):
    """Load products from Excel (cached)."""
    try:
        products = load_startech_products(excel_path)
        stats = get_product_statistics(products)
        return products, stats, None
    except FileNotFoundError:
        return [], {}, f"File not found: {excel_path}"
    except Exception as e:
        return [], {}, f"Error loading Excel: {str(e)}"


def get_components(products) -> OrchestratorComponents:
    """Initialize ST-Bot components (cached via products)."""

    # Create search function that filters products
    def product_search_func(filter_dict):
        """Filter products based on criteria."""
        results = []
        for product in products:
            match = True

            # Category filter
            if 'category' in filter_dict and filter_dict['category']:
                product_cat = product.metadata.get('category', '').lower()
                search_cat = filter_dict['category'].lower()
                if search_cat.endswith('s'):
                    search_cat = search_cat[:-1]
                if product_cat.endswith('s'):
                    product_cat = product_cat[:-1]
                if product_cat != search_cat:
                    match = False

            # Connector filters
            if match and 'connector_from' in filter_dict and filter_dict['connector_from']:
                connectors = product.metadata.get('connectors', [])
                if connectors and len(connectors) >= 1:
                    source_lower = str(connectors[0]).lower()
                    search_term = filter_dict['connector_from'].lower()
                    # Handle USB-C variations
                    if 'usb-c' in search_term or 'usb c' in search_term or 'type-c' in search_term:
                        if not any(v in source_lower for v in ['usb-c', 'usb c', 'type-c', 'type c', 'usb type-c']):
                            match = False
                    elif search_term not in source_lower:
                        match = False
                else:
                    match = False

            if match and 'connector_to' in filter_dict and filter_dict['connector_to']:
                connectors = product.metadata.get('connectors', [])
                if connectors and len(connectors) >= 2:
                    target_lower = str(connectors[1]).lower()
                    search_term = filter_dict['connector_to'].lower()
                    if search_term not in target_lower:
                        match = False
                    # For same-connector cables (HDMI-to-HDMI), verify both ends match
                    if match and filter_dict.get('same_connector'):
                        source_lower = str(connectors[0]).lower()
                        if search_term not in source_lower:
                            match = False
                else:
                    match = False

            # Length filter
            if match and 'length' in filter_dict and filter_dict['length']:
                product_length_ft = product.metadata.get('length_ft')
                if product_length_ft:
                    requested_ft = filter_dict['length']
                    length_unit = filter_dict.get('length_unit', 'ft')
                    length_pref = filter_dict.get('length_preference', LengthPreference.EXACT_OR_LONGER)
                    # Convert to feet based on unit
                    if length_unit == 'm':
                        requested_ft = filter_dict['length'] * 3.28084
                    elif length_unit == 'in':
                        requested_ft = filter_dict['length'] / 12.0  # 12 inches per foot
                    elif length_unit == 'cm':
                        requested_ft = filter_dict['length'] / 30.48  # 30.48 cm per foot

                    # Apply length preference
                    if length_pref == LengthPreference.EXACT_OR_SHORTER:
                        # "under X", "up to X" - only match products <= requested length
                        # Use small tolerance for rounding (0.1ft = ~1 inch)
                        if product_length_ft > requested_ft + 0.1:
                            match = False
                    elif length_pref == LengthPreference.EXACT_OR_LONGER:
                        # Default - match products >= requested length (with tolerance)
                        tolerance_ft = max(0.5, requested_ft * 0.1)
                        if product_length_ft < requested_ft - tolerance_ft:
                            match = False
                    else:
                        # CLOSEST - use tolerance around requested length
                        tolerance_ft = max(0.5, requested_ft * 0.2)
                        if abs(product_length_ft - requested_ft) > tolerance_ft:
                            match = False

            # Color filter
            if match and 'color' in filter_dict and filter_dict['color']:
                product_color = product.metadata.get('color', '').lower()
                requested_color = filter_dict['color'].lower()
                if requested_color not in product_color:
                    match = False

            if match:
                results.append(product)

        return results

    # Create wrapper for SearchStrategy
    class SearchEngineWrapper:
        def __init__(self, strategy, search_func):
            self.strategy = strategy
            self.search_func = search_func

        def search(self, filters):
            return self.strategy.search(filters, self.search_func)

    search_engine = SearchEngineWrapper(SearchStrategy(), product_search_func)

    return OrchestratorComponents(
        intent_classifier=IntentClassifier(),
        filter_extractor=FilterExtractor(),
        search_engine=search_engine,
        product_ranker=ProductRanker(),
        response_builder=ResponseBuilder(),
        formatter=ResponseFormatter(),
        query_analyzer=QueryAnalyzer(),
        logger=ConversationLogger("startech_conversations.csv"),
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.title("🤖 ST-Bot - StarTech.com Product Assistant")
    st.markdown("*Powered by your 4,178+ product catalog*")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        excel_path = st.text_input(
            "📁 Excel File Path",
            value="Main Data AI Bot.xlsx",
            help="Path to your Excel file"
        )
        st.markdown("---")

    # Load products
    products, stats, error = load_products(excel_path)

    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please ensure Main_Data_AI_Bot.xlsx is in the same folder as this app")
        st.stop()

    if not products:
        st.warning("⚠️ No products loaded. Check your Excel file.")
        st.stop()

    # Store products in session state
    st.session_state._all_products = products

    # Sidebar - Product Statistics
    with st.sidebar:
        st.header("📦 Product Catalog")
        st.metric("Total Products", stats['total'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("With Length", stats['with_length'])
        with col2:
            st.metric("With Connectors", stats['with_connectors'])

        with st.expander("📊 Categories"):
            for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True)[:10]:
                st.write(f"• **{cat.title()}:** {count}")

        st.markdown("---")

    # Initialize session state
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
        st.session_state.context = ConversationContext()
        st.session_state.messages = []
        st.session_state.pending_guidance_data = None
        st.session_state.pending_question_data = None

    # Get components
    components = get_components(products)

    # Sidebar - Session Stats
    with st.sidebar:
        st.header("📊 Session Stats")
        st.write(f"**Session ID:** `{st.session_state.session.session_id[:16]}...`")
        st.write(f"**Messages:** {st.session_state.session.get_message_count()}")

        if st.session_state.context.current_products:
            st.write(f"**Products in Context:** {len(st.session_state.context.current_products)}")

        if st.button("🔄 New Session"):
            st.session_state.session = SessionState()
            st.session_state.context = ConversationContext()
            st.session_state.messages = []
            st.session_state.pending_guidance_data = None
            st.session_state.pending_question_data = None
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("What StarTech.com product are you looking for?")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response using orchestrator
        with st.chat_message("assistant"):
            with st.spinner("Searching 4,000+ products..."):
                response, intent_type = process_query(
                    query=prompt,
                    context=st.session_state.context,
                    session=st.session_state.session,
                    components=components,
                    all_products=products,
                    st_session_state=st.session_state,
                    debug_mode=DEBUG_MODE
                )
                st.markdown(response)

                # Debug info
                if DEBUG_MODE:
                    with st.expander("🔍 Debug Info"):
                        st.write(f"**Intent Detected:** {intent_type}")
                        st.write(f"**Total Products Available:** {len(products)}")
                        if st.session_state.context.current_products:
                            st.write(f"**Products in Context:** {len(st.session_state.context.current_products)}")

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save to session
        st.session_state.session.add_message("user", prompt)
        st.session_state.session.add_message("assistant", response)


if __name__ == "__main__":
    main()
