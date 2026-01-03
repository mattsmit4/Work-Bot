"""
Core data models for ST-Bot.

Defines all data structures used throughout the application.
These are pure Python dataclasses with no external dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
from datetime import datetime


class IntentType(Enum):
    """
    User intent types - simplified MVP with 5 core types.

    Priority order:
    1. GREETING - Simple greetings
    2. FAREWELL - Goodbye messages
    3. NEW_SEARCH - New product search
    4. FOLLOWUP - Questions about products in context
    5. AMBIGUOUS - Can't determine intent
    """
    GREETING = "greeting"
    FAREWELL = "farewell"
    NEW_SEARCH = "new_search"
    FOLLOWUP = "followup"  # Combines all followup types
    AMBIGUOUS = "ambiguous"


@dataclass
class Intent:
    """
    User intent with metadata.
    
    Attributes:
        type: Intent classification
        confidence: Confidence score (0.0-1.0)
        reasoning: Why this intent was selected
        sku: Product SKU if explicit_sku intent
        meta_info: Additional metadata for intent
    """
    type: IntentType
    confidence: float
    reasoning: str
    sku: Optional[str] = None
    meta_info: Optional[dict] = None
    
    def __str__(self) -> str:
        return f"Intent({self.type.value}, confidence={self.confidence:.2f})"


@dataclass
class Product:
    """
    Product information.
    
    Attributes:
        product_number: StarTech product SKU
        content: Full product specification text
        metadata: Product metadata (category, subcategory, etc.)
        score: Relevance score from search
    """
    product_number: str
    content: str
    metadata: dict[str, Any]
    score: float = 1.0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value safely."""
        return self.metadata.get(key, default)

    def supports_4k(self) -> bool:
        """
        Check if product supports 4K resolution.

        Checks ALL possible 4K indicators for consistency across the codebase:
        1. DOCK4KSUPPORT field (for docks)
        2. features list contains '4K'
        3. max_resolution / MAXRESOLUTION / max_dvi_resolution contains 4K indicators
        4. product content contains '4k'
        5. Inherent capability based on cable/connector type (HDMI, DisplayPort, Thunderbolt)

        Returns:
            True if product supports 4K, False otherwise
        """
        meta = self.metadata
        content_lower = (self.content or '').lower()

        # Check 1: DOCK4KSUPPORT field (most authoritative for docks)
        dock_4k = meta.get('DOCK4KSUPPORT', '')
        if dock_4k and str(dock_4k).lower() not in ('no', '', 'nan'):
            return True

        # Check 2: features list
        features = meta.get('features', [])
        if any('4k' in str(f).lower() for f in features):
            return True

        # Check 3: Resolution fields (max_resolution, MAXRESOLUTION, max_dvi_resolution)
        resolution_fields = ['max_resolution', 'MAXRESOLUTION', 'max_dvi_resolution']
        for field in resolution_fields:
            res_value = meta.get(field, '')
            if res_value:
                res_str = str(res_value).lower()
                # Check for 4K indicators: '4k', '2160', '3840', 'uhd', 'ultra hd'
                if any(ind in res_str for ind in ('4k', '2160', '3840', 'uhd', 'ultra hd')):
                    return True

        # Check 4: Content fallback (least reliable but catches edge cases)
        if '4k' in content_lower:
            return True

        # Check 5: Inherent capability based on cable/connector type
        # HDMI, DisplayPort, and Thunderbolt cables inherently support 4K
        if self._has_inherent_4k_capability():
            return True

        return False

    def _has_inherent_4k_capability(self) -> bool:
        """
        Check if product type inherently supports 4K resolution.

        Cable types that inherently support 4K:
        - HDMI cables (High Speed HDMI 1.4+ supports 4K@30Hz, HDMI 2.0+ supports 4K@60Hz)
        - DisplayPort cables (DP 1.2+ supports 4K@60Hz)
        - Thunderbolt cables (TB3/TB4 supports 4K and higher)

        Returns:
            True if cable type inherently supports 4K
        """
        meta = self.metadata
        content_lower = (self.content or '').lower()
        name_lower = meta.get('name', '').lower()
        connectors = meta.get('connectors', [])
        connector_str = ' '.join(str(c).lower() for c in connectors)
        sub_category = (meta.get('sub_category', '') or '').lower()
        category = (meta.get('category', '') or '').lower()

        # Check if this is a cable/adapter product
        is_cable = category in ('cable', 'adapter') or meta.get('length_ft') or 'cable' in sub_category

        if not is_cable:
            return False

        # VGA is analog and can't do 4K - if VGA is involved, no inherent 4K
        if 'vga' in connector_str or 'vga' in name_lower or 'vga' in sub_category:
            return False

        # DisplayPort cables inherently support 4K
        if 'displayport' in connector_str or 'displayport' in name_lower or 'dp cable' in sub_category:
            return True

        # HDMI cables inherently support 4K (HDMI 1.4+ at 30Hz, HDMI 2.0+ at 60Hz)
        if 'hdmi' in connector_str or 'hdmi' in name_lower or 'hdmi' in sub_category:
            return True

        # Thunderbolt cables inherently support 4K
        if 'thunderbolt' in connector_str or 'thunderbolt' in name_lower or 'thunderbolt' in sub_category:
            return True

        return False

    def supports_resolution(self, resolution: str) -> bool:
        """
        Check if product supports a specific resolution.

        Args:
            resolution: Resolution to check ('4k', '8k', '1440p', '1080p')

        Returns:
            True if product supports the resolution
        """
        res_lower = resolution.lower()

        if res_lower == '4k':
            return self.supports_4k()

        meta = self.metadata
        content_lower = (self.content or '').lower()
        features = meta.get('features', [])

        # Resolution indicators
        resolution_indicators = {
            '8k': ['8k', '4320', '7680'],
            '1440p': ['1440p', '1440', '2560', 'qhd', '2k'],
            '1080p': ['1080p', '1080', '1920', 'full hd', 'fhd'],
        }

        indicators = resolution_indicators.get(res_lower, [res_lower])

        # Check features list
        for f in features:
            f_lower = str(f).lower()
            if any(ind in f_lower for ind in indicators):
                return True

        # Check resolution fields
        resolution_fields = ['max_resolution', 'MAXRESOLUTION', 'max_dvi_resolution']
        for field in resolution_fields:
            res_value = meta.get(field, '')
            if res_value:
                res_str = str(res_value).lower()
                if any(ind in res_str for ind in indicators):
                    return True

        # Check content
        if any(ind in content_lower for ind in indicators):
            return True

        # Check inherent capability for common resolutions
        # If a cable supports 4K, it also supports 1080p and 1440p
        if res_lower in ('1080p', '1440p'):
            if self._has_inherent_4k_capability():
                return True

        return False


class GuidancePhase(Enum):
    """Phases of a guidance conversation."""
    INITIAL_QUESTIONS = "initial_questions"  # Asking about ports/inputs
    PORT_COUNT_CLARIFICATION = "port_count_clarification"  # Asking how many of each port
    READY_TO_RECOMMEND = "ready_to_recommend"  # Have all info, can make recommendation
    OFFERED_DOCK = "offered_dock"  # Offered to show docks, waiting for yes/no
    COMPLETE = "complete"  # Recommendation given


@dataclass
class PendingGuidance:
    """
    Tracks pending setup guidance that's awaiting user input.

    Attributes:
        setup_type: Type of setup (e.g., 'multi_monitor', 'dock_selection')
        monitor_count: Number of monitors (if applicable)
        phase: Current phase of the guidance conversation
        computer_ports: Port types the user's computer has (e.g., ['USB-C', 'HDMI'])
        computer_port_counts: How many of each port (e.g., {'USB-C': 2, 'HDMI': 1})
        monitor_inputs: Input types for each monitor (e.g., ['HDMI', 'DisplayPort', 'VGA'])
        preference: User's preference (e.g., 'individual_cables', 'dock')

        # Dock-specific fields
        dock_use_cases: What user needs dock for (e.g., ['monitors', 'charging', 'ports'])
        dock_laptop_port: Primary port on user's laptop (e.g., 'USB-C', 'Thunderbolt')
        dock_must_have_features: Required features (e.g., ['power_delivery', 'ethernet'])
    """
    setup_type: str
    monitor_count: Optional[int] = None
    phase: GuidancePhase = GuidancePhase.INITIAL_QUESTIONS
    computer_ports: list[str] = field(default_factory=list)
    computer_port_counts: dict[str, int] = field(default_factory=dict)
    monitor_inputs: list[str] = field(default_factory=list)
    preference: Optional[str] = None  # 'individual_cables', 'dock', 'hub'

    # Dock-specific fields
    dock_use_cases: list[str] = field(default_factory=list)
    dock_laptop_port: Optional[str] = None
    dock_must_have_features: list[str] = field(default_factory=list)

    # KVM-specific fields
    kvm_port_count: Optional[int] = None  # 2, 4, 8, 16 computers
    kvm_video_type: Optional[str] = None  # HDMI, DisplayPort, VGA, DVI
    kvm_usb_switching: Optional[bool] = None  # Do they need USB device switching?
    kvm_features: list[str] = field(default_factory=list)  # audio, hotkey, 4K, etc.

    # Single monitor connection fields
    cable_length: Optional[str] = None  # e.g., "6 ft", "long", "short"

    # Assumption tracking (for "answer first, clarify if needed" UX)
    port_assumption_made: bool = False  # True if we assumed port counts

    @property
    def awaiting_answers(self) -> bool:
        """Check if we're still waiting for user input."""
        # OFFERED_DOCK is also awaiting - waiting for yes/no to dock offer
        return self.phase in [
            GuidancePhase.INITIAL_QUESTIONS,
            GuidancePhase.PORT_COUNT_CLARIFICATION,
            GuidancePhase.OFFERED_DOCK
        ]

    @property
    def total_ports(self) -> int:
        """Get total number of video output ports."""
        if self.computer_port_counts:
            return sum(self.computer_port_counts.values())
        # If no counts specified, assume 1 of each type
        return len(self.computer_ports)

    def needs_port_clarification(self) -> bool:
        """Check if we need to ask about port counts.

        NOTE: We no longer ask for clarification. Instead, we assume the user
        has enough ports and provide recommendations immediately, with a note
        about the assumption and a dock fallback option.

        This method always returns False to prevent UX friction.
        """
        # Never ask for port count clarification - answer first, clarify if needed
        return False


class PendingQuestionType(Enum):
    """Types of questions the bot can ask that need follow-up tracking."""
    DAISY_CHAIN_DP_CHECK = "daisy_chain_dp_check"  # "Do your monitors have DisplayPort inputs AND outputs?"


@dataclass
class PendingQuestion:
    """
    Tracks a question the bot asked that awaits a user response.

    Attributes:
        question_type: Type of question asked
        context_data: Any data needed to process the answer (e.g., original query)
    """
    question_type: PendingQuestionType
    context_data: dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """
    Conversation state for intent classification.

    This replaces the scattered st.session_state variables with
    a clean, structured context object.

    Attributes:
        current_products: List of products currently shown to user
        last_product: Single product currently in context
        last_filters: Filters used in last search
        query_count: Number of queries in session
        session_id: Unique session identifier
        daisy_chain_query: Whether last query was about daisy-chaining
        pending_guidance: Active guidance flow awaiting user answers
        pending_question: Question the bot asked that awaits user response
    """
    current_products: Optional[list[Product]] = None
    last_product: Optional[Product] = None
    last_filters: Optional[dict] = None
    query_count: int = 0
    session_id: Optional[str] = None
    daisy_chain_query: bool = False
    pending_guidance: Optional[PendingGuidance] = None
    pending_question: Optional[PendingQuestion] = None
    last_comparison_indices: Optional[list[int]] = None  # Track which products were just compared
    pending_feature_search: Optional[dict] = None  # Track offered feature search (e.g., {'feature': '4K', 'product_type': 'HDMI cables'})

    def has_multi_product_context(self) -> bool:
        """Check if user is viewing multiple products."""
        return bool(self.current_products)

    def has_single_product_context(self) -> bool:
        """Check if user is viewing a single product."""
        return bool(self.last_product)

    def has_pending_guidance(self) -> bool:
        """Check if there's an active guidance flow awaiting answers."""
        return self.pending_guidance is not None and self.pending_guidance.awaiting_answers

    def start_guidance(self, setup_type: str, monitor_count: Optional[int] = None) -> None:
        """Start a new guidance flow."""
        # Clear any pending educational question to avoid conflicts
        # Setup guidance takes priority - educational questions can be re-asked later
        self.pending_question = None
        self.pending_guidance = PendingGuidance(
            setup_type=setup_type,
            monitor_count=monitor_count,
            phase=GuidancePhase.INITIAL_QUESTIONS
        )

    def complete_guidance(self) -> None:
        """Mark guidance as complete (answers received)."""
        if self.pending_guidance:
            self.pending_guidance.phase = GuidancePhase.COMPLETE

    def clear_guidance(self) -> None:
        """Clear pending guidance."""
        self.pending_guidance = None

    def clear_products(self) -> None:
        """Clear all product context."""
        self.current_products = None
        self.last_product = None

    def set_multi_products(self, products: list[Product]) -> None:
        """Set multiple products as current context."""
        self.current_products = products
        self.last_product = None  # Clear single product

    def set_single_product(self, product: Product) -> None:
        """Set single product as current context."""
        self.last_product = product
        self.current_products = None  # Clear multi products

    def has_pending_question(self) -> bool:
        """Check if there's a pending question awaiting user response."""
        return self.pending_question is not None

    def set_pending_question(
        self,
        question_type: PendingQuestionType,
        context_data: Optional[dict] = None
    ) -> None:
        """Set a pending question that awaits user response."""
        self.pending_question = PendingQuestion(
            question_type=question_type,
            context_data=context_data or {}
        )

    def clear_pending_question(self) -> None:
        """Clear the pending question."""
        self.pending_question = None

    def set_comparison_context(self, indices: list[int]) -> None:
        """Track which products were just compared."""
        self.last_comparison_indices = indices

    def clear_comparison_context(self) -> None:
        """Clear comparison context (e.g., when new search happens)."""
        self.last_comparison_indices = None

    def has_comparison_context(self) -> bool:
        """Check if there's a recent comparison context."""
        return self.last_comparison_indices is not None and len(self.last_comparison_indices) >= 2

    def get_compared_products(self) -> Optional[list[Product]]:
        """Get the products that were just compared, if any."""
        if not self.has_comparison_context() or not self.current_products:
            return None
        result = []
        for idx in self.last_comparison_indices:
            if 1 <= idx <= len(self.current_products):
                result.append(self.current_products[idx - 1])
        return result if len(result) >= 2 else None

    def set_pending_feature_search(
        self,
        feature: str,
        product_type: str,
        connector_from: Optional[str] = None,
        connector_to: Optional[str] = None,
        category: Optional[str] = None
    ) -> None:
        """Track that we offered to search for a feature.

        Args:
            feature: The feature user wants (e.g., "4K")
            product_type: User-friendly description (e.g., "HDMI cables")
            connector_from: Source connector type (e.g., "HDMI")
            connector_to: Destination connector type (e.g., "HDMI")
            category: Product category (e.g., "Cables")
        """
        self.pending_feature_search = {
            'feature': feature,
            'product_type': product_type,
            'connector_from': connector_from,
            'connector_to': connector_to,
            'category': category
        }

    def has_pending_feature_search(self) -> bool:
        """Check if there's a pending feature search offer."""
        return self.pending_feature_search is not None

    def clear_pending_feature_search(self) -> None:
        """Clear the pending feature search offer."""
        self.pending_feature_search = None


@dataclass
class LLMQueryIntent:
    """
    Structured output from LLM query understanding.
    
    Attributes:
        product_type: Type of product (cable, dock, adapter, etc.)
        confidence: LLM confidence in interpretation
        reasoning: LLM's explanation of interpretation
        requirements: Extracted requirements as dict
        ambiguity: Clarification question if query is unclear
    """
    product_type: str
    confidence: float
    reasoning: str
    requirements: dict[str, Any]
    ambiguity: Optional[str] = None


class LengthPreference(Enum):
    """User preference for length alternatives when exact match unavailable."""
    EXACT_OR_LONGER = "exact_or_longer"  # Default: prefer next size up
    EXACT_OR_SHORTER = "exact_or_shorter"  # User prefers shorter
    CLOSEST = "closest"  # User is flexible, show closest either direction


@dataclass
class DroppedFilter:
    """
    Information about a filter that was relaxed during search.

    Attributes:
        filter_name: Name of the filter (e.g., "length", "features")
        requested_value: What the user asked for
        reason: Why it was dropped
        alternatives: Available alternatives for this filter
    """
    filter_name: str
    requested_value: Any
    reason: str
    alternatives: Optional[list[Any]] = None


@dataclass
class SearchResult:
    """
    Search result with products and metadata.

    Attributes:
        products: List of products found
        filters_used: Filters that were applied
        tier: Which search tier succeeded (1-5, None=semantic only)
        total_count: Total number of results
        original_filters: Original filters from user request
        dropped_filters: Filters that were relaxed with explanations
    """
    products: list[Product]
    filters_used: dict
    tier: Optional[int] = None
    total_count: int = 0
    original_filters: Optional[dict] = None
    dropped_filters: list[DroppedFilter] = field(default_factory=list)
    category_relaxed: bool = False  # True if we swapped cable→adapter

    def had_filter_relaxation(self) -> bool:
        """Check if any filters were relaxed."""
        return len(self.dropped_filters) > 0

    def get_dropped_filter(self, name: str) -> Optional[DroppedFilter]:
        """Get a specific dropped filter by name."""
        for df in self.dropped_filters:
            if df.filter_name == name:
                return df
        return None


@dataclass
class ConversationLog:
    """
    Single conversation log entry.
    
    Attributes:
        timestamp: When the interaction occurred
        session_id: Session identifier
        query_number: Query count in session
        user_message: What user asked
        bot_response: What bot replied
        products_shown: Products displayed to user
        intent_type: Classified intent
        filters_applied: Filters used in search
        match_status: Result status (success, no-match, other)
    """
    timestamp: datetime
    session_id: str
    query_number: int
    user_message: str
    bot_response: str
    products_shown: list[dict] = field(default_factory=list)
    intent_type: Optional[str] = None
    filters_applied: dict = field(default_factory=dict)
    match_status: str = "other"


@dataclass
class FilterConfig:
    """
    Configuration for filter extraction.
    
    Attributes:
        categorical_values: Available categories, subcategories, etc.
        sku_set: Set of valid product SKUs
        sku_map: Mapping of SKUs without hyphens to canonical form
    """
    categorical_values: dict[str, list]
    sku_set: set[str]
    sku_map: dict[str, str]


@dataclass
class SearchFilters:
    """
    Extracted search filters from user query.

    Attributes:
        length: Length requirement (e.g., 6.0)
        length_unit: Unit of length (e.g., "ft", "m")
        length_preference: How to handle length alternatives (default: next size up)
        connector_from: Source connector type (e.g., "USB-C")
        connector_to: Destination connector type (e.g., "HDMI")
        features: Technical features (e.g., ["4K", "Thunderbolt"])
        product_category: Product category (e.g., "Cables", "Adapters")
        port_count: Number of ports (for hubs, switches, etc.)
        color: Requested product color (e.g., "Black", "White", "Red")
        keywords: Search keywords for text matching (e.g., ["fiber", "optic", "patch"])
        required_port_types: Port types the product must have (e.g., ["USB-C"] for docks)
        min_monitors: Minimum number of monitors the dock must support
    """
    length: Optional[float] = None
    length_unit: Optional[str] = None
    length_preference: LengthPreference = LengthPreference.EXACT_OR_LONGER
    connector_from: Optional[str] = None
    connector_to: Optional[str] = None
    features: list[str] = field(default_factory=list)
    product_category: Optional[str] = None
    port_count: Optional[int] = None
    color: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    required_port_types: list[str] = field(default_factory=list)
    min_monitors: Optional[int] = None