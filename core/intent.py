"""
Intent classification for ST-Bot - Simplified MVP.

Determines user intent from their message and conversation context.
Uses simple, reliable rules for 5 core intent types.
"""

import re
from core.context import Intent, IntentType, ConversationContext
from core.structured_logging import get_logger
from config.patterns import GREETING_PATTERNS, FAREWELL_PATTERNS, has_pattern
from config.synonyms import expand_synonyms

# Module-level logger
_logger = get_logger("core.intent")


class IntentClassifier:
    """
    Classifies user intent - Simplified MVP.

    Only 5 intent types:
    1. GREETING - Simple greetings
    2. FAREWELL - Goodbye messages
    3. NEW_SEARCH - New product search
    4. FOLLOWUP - Questions about products in context
    5. AMBIGUOUS - Can't determine intent

    Example:
        classifier = IntentClassifier()
        context = ConversationContext()
        intent = classifier.classify("Hello!", context)
        # Returns: Intent(type=GREETING, confidence=1.0, ...)
    """

    def __init__(self):
        """Initialize the intent classifier."""
        pass

    def classify(self, prompt: str, context: ConversationContext) -> Intent:
        """
        Classify user intent.

        Args:
            prompt: User's message
            context: Conversation context (previous products, filters, etc.)

        Returns:
            Intent object with type, confidence, and reasoning
        """
        prompt_lower = prompt.lower()
        prompt_expanded = expand_synonyms(prompt)
        word_count = len(prompt.split())

        _logger.debug(
            "Classifying intent",
            extra={
                "event": "intent_classify_start",
                "word_count": word_count,
                "has_product_context": context.has_multi_product_context() or context.has_single_product_context(),
            }
        )

        # Priority 1: Greetings (short messages only)
        if self._is_greeting(prompt_lower, word_count):
            return Intent(
                type=IntentType.GREETING,
                confidence=1.0,
                reasoning="User sent a greeting"
            )

        # Priority 2: Farewells
        if self._is_farewell(prompt_lower):
            return Intent(
                type=IntentType.FAREWELL,
                confidence=1.0,
                reasoning="User is ending the conversation"
            )

        # Priority 3: Check for direct SKU lookup (before context-based checks)
        # SKU lookups are very specific - if someone types a SKU, they want that product
        sku_match = self._extract_sku(prompt)
        if sku_match:
            return Intent(
                type=IntentType.NEW_SEARCH,
                confidence=0.95,
                reasoning=f"User is looking up product SKU: {sku_match}",
                sku=sku_match
            )

        # Priority 4: Check if user has product context
        has_context = context.has_multi_product_context() or context.has_single_product_context()

        # If user has product context, determine if this is a followup or new search
        if has_context:
            # FIRST: Check for explicit new search patterns
            # "Show me DisplayPort cables" is always a new search, even with context
            if self._is_explicit_new_search(prompt_lower):
                return Intent(
                    type=IntentType.NEW_SEARCH,
                    confidence=0.9,
                    reasoning="User explicitly requested new product search"
                )

            # Check if query mentions a DIFFERENT connector type than context
            if self._has_different_connector(prompt_lower, context):
                return Intent(
                    type=IntentType.NEW_SEARCH,
                    confidence=0.9,
                    reasoning="User specified different connector type"
                )

            # Check if query mentions a DIFFERENT product category than context
            if self._has_different_category(prompt_lower, context):
                return Intent(
                    type=IntentType.NEW_SEARCH,
                    confidence=0.9,
                    reasoning="User specified different product category"
                )

            # Check for refinement (e.g., "I need a 10ft version instead")
            if self._is_constraint_refinement(prompt_lower):
                return Intent(
                    type=IntentType.FOLLOWUP,
                    confidence=0.9,
                    reasoning="User wants to refine current results",
                    meta_info={'refinement': True}
                )

            # Check if this looks like a new search (has product keywords/connectors)
            if self._is_new_search(prompt_lower, prompt_expanded):
                return Intent(
                    type=IntentType.NEW_SEARCH,
                    confidence=0.9,
                    reasoning="User specified new product criteria"
                )

            # Check if this is a followup question
            if self._is_followup(prompt_lower, word_count):
                return Intent(
                    type=IntentType.FOLLOWUP,
                    confidence=0.9,
                    reasoning="User is asking about products in context"
                )

        # Priority 5: New product search
        if self._has_domain_tokens(prompt_expanded):
            return Intent(
                type=IntentType.NEW_SEARCH,
                confidence=0.8,
                reasoning="User is searching for products"
            )

        # Fallback: Use context if available
        if has_context:
            return Intent(
                type=IntentType.FOLLOWUP,
                confidence=0.6,
                reasoning="Unclear query with product context"
            )

        # Ultimate fallback
        return Intent(
            type=IntentType.AMBIGUOUS,
            confidence=0.5,
            reasoning="Could not determine user intent"
        )

    # === Detection Methods ===

    def _is_greeting(self, text: str, word_count: int) -> bool:
        """Check if text is a greeting (max 4 words, no product keywords)."""
        if word_count > 4:
            return False
        # Not a greeting if it contains product-related words
        # "Hi, I need cables" should be NEW_SEARCH, not GREETING
        product_keywords = r'\b(?:cable|cables|adapter|adapters|dock|docks|hub|hubs|' \
                          r'hdmi|displayport|usb|thunderbolt|tb3|tb4|ethernet|monitor|' \
                          r'need|looking|find|show|want)\b'
        if re.search(product_keywords, text):
            return False
        return has_pattern(text, GREETING_PATTERNS)

    def _is_farewell(self, text: str) -> bool:
        """Check if text is a farewell."""
        return has_pattern(text, FAREWELL_PATTERNS)

    def _is_new_search(self, text: str, text_expanded: str) -> bool:
        """
        Check if this looks like a new product search.

        Returns True if query has product search indicators like:
        - Connector types (HDMI, USB-C, DisplayPort)
        - Product types (cable, adapter, dock)
        - Length specifications (6ft, 10 foot)
        - "show me", "find me" patterns
        """
        # Explicit search request patterns
        search_patterns = [
            r'\b(?:show|find|get|list|give)\s+me\b',
            r'\blooking\s+for\b',
            r'\bneed\s+(?:a|an|some)\b',
            r'\bwant\s+(?:a|an|some)\b',
        ]
        if any(re.search(pat, text) for pat in search_patterns):
            return True

        # Connector-to-connector pattern (e.g., "USB-C to HDMI")
        if re.search(
            r'\b(?:usb-?c|type-?c|displayport|hdmi|thunderbolt|vga|dvi)\s+'
            r'to\s+'
            r'(?:usb-?c|type-?c|displayport|hdmi|vga|dvi)\b',
            text
        ):
            return True

        # Product type with connector (e.g., "HDMI cable", "USB-C adapter")
        if re.search(
            r'\b(?:hdmi|displayport|usb-?c?|thunderbolt|vga|dvi|ethernet)\s*'
            r'(?:to\s+(?:hdmi|displayport|usb-?c?|vga|dvi))?\s*'
            r'(?:cable|adapter|converter|cord)\b',
            text
        ):
            return True

        # Length + product type (e.g., "6ft HDMI cable")
        has_length = bool(re.search(
            r'\b\d+(?:\.\d+)?\s*(?:ft|foot|feet|m|meter|meters)\b',
            text
        ))
        if has_length and self._has_domain_tokens(text_expanded):
            return True

        return False

    def _is_followup(self, text: str, word_count: int) -> bool:
        """
        Check if this is a follow-up question about products in context.
        """
        # References to products in context
        followup_patterns = [
            r'\b(?:these|them|those|it|this|that)\b',
            r'\bwhich\s+(?:one|product|item)\b',
            r'\bthe\s+(?:shortest|longest|cheapest|best|first|second|third)\b',
            r'\btell\s+me\s+(?:more|about)\b',
            r'\b(?:does|do|can|will|is)\s+(?:this|it|the)\b',
            r'\b(?:difference|compare|between)\b',
            r'\bproduct\s*[123]\b',
            r'\b#[123]\b',
        ]
        if any(re.search(pat, text) for pat in followup_patterns):
            return True

        # Short queries with context are likely followups
        if word_count <= 6:
            return True

        return False

    def _is_explicit_new_search(self, text: str) -> bool:
        """
        Check if query is an explicit new search request with a connector type.

        Patterns like "show me DisplayPort cables" are always new searches.
        But "I need a 10ft cable instead" is a refinement (no connector specified).
        """
        # Must have an explicit search phrase
        has_search_phrase = bool(re.search(
            r'\b(?:show|find|get|list)\s+(?:me\s+)?',
            text
        ))

        if not has_search_phrase:
            return False

        # Must mention a specific connector type (not just "cable")
        has_connector = bool(re.search(
            r'\b(?:hdmi|displayport|dp|usb[- ]?c|type[- ]?c|usb[- ]?a|vga|dvi|thunderbolt|tb3|tb4|ethernet)\b',
            text
        ))

        # Must mention a product type
        has_product_type = bool(re.search(
            r'\b(?:cable|cables|adapter|adapters|dock|docks|hub|hubs)\b',
            text
        ))

        return has_connector and has_product_type

    def _has_different_connector(self, text: str, context: ConversationContext) -> bool:
        """
        Check if query mentions a different connector type than what's in context.

        This prevents "DisplayPort cables under 6ft" from being treated as a
        refinement when context has USB-C to HDMI products.
        """
        # Connector types to check (order matters - check more specific first)
        connector_patterns = {
            'displayport': r'\b(?:displayport|dp)\b',
            'hdmi': r'\bhdmi\b',
            'usb-c': r'\b(?:usb[- ]?c|type[- ]?c)\b',
            'usb-a': r'\b(?:usb[- ]?a|type[- ]?a)\b',
            'vga': r'\bvga\b',
            'dvi': r'\bdvi\b',
            'thunderbolt': r'\b(?:thunderbolt|tb3|tb4)\b',
            'ethernet': r'\b(?:ethernet|cat[56]e?|rj-?45)\b',
        }

        # Find connectors mentioned in query
        query_connectors = set()
        for name, pattern in connector_patterns.items():
            if re.search(pattern, text):
                query_connectors.add(name)

        if not query_connectors:
            return False  # No connectors in query

        # Get PRIMARY connectors from context products
        # Only look at the main connector type, not modes like "DisplayPort Alt Mode"
        context_connectors = set()
        if context.current_products:
            for product in context.current_products:
                connectors = product.metadata.get('connectors', [])
                for conn in connectors:
                    conn_lower = conn.lower()
                    # Extract primary connector - skip if it's just a mode descriptor
                    # "1 x USB-C (24 pin) DisplayPort Alt Mode" → USB-C (not DisplayPort)
                    # "1 x DisplayPort" → DisplayPort
                    if 'alt mode' in conn_lower or 'alternate mode' in conn_lower:
                        # This is a mode descriptor, extract the actual connector
                        if re.search(r'\busb[- ]?c\b|type[- ]?c\b', conn_lower):
                            context_connectors.add('usb-c')
                        continue

                    # Normal connector - check patterns
                    for name, pattern in connector_patterns.items():
                        if re.search(pattern, conn_lower):
                            context_connectors.add(name)
                            break  # Only add one connector type per connector string

        if not context_connectors:
            return False  # No connector info in context

        # If query has connectors NOT in context, it's a different search
        new_connectors = query_connectors - context_connectors
        return len(new_connectors) > 0

    def _has_different_category(self, text: str, context: ConversationContext) -> bool:
        """
        Check if query mentions a different product category than what's in context.

        This prevents "USB hub" from being treated as a followup when context has docks.
        """
        # Product category patterns - maps category names to detection patterns
        category_patterns = {
            'cables': r'\b(?:cable|cables|cord|cords)\b',
            'adapters': r'\b(?:adapter|adapters|converter|converters)\b',
            'docks': r'\b(?:dock|docks|docking)\b',
            'hubs': r'\b(?:hub|hubs)\b',
            'kvm': r'\b(?:kvm)\b',
            'switches': r'\b(?:switch|switches)\b',
            'mounts': r'\b(?:mount|mounts|stand|stands)\b',
            'enclosures': r'\b(?:enclosure|enclosures)\b',
            'splitters': r'\b(?:splitter|splitters)\b',
        }

        # Find categories mentioned in query
        query_categories = set()
        for name, pattern in category_patterns.items():
            if re.search(pattern, text):
                query_categories.add(name)

        if not query_categories:
            return False  # No category in query

        # Get categories from context products
        context_categories = set()
        if context.current_products:
            for product in context.current_products:
                cat = product.metadata.get('category', '').lower()
                if cat:
                    # Normalize category names
                    if 'dock' in cat:
                        context_categories.add('docks')
                    elif 'hub' in cat:
                        context_categories.add('hubs')
                    elif 'cable' in cat:
                        context_categories.add('cables')
                    elif 'adapter' in cat:
                        context_categories.add('adapters')
                    elif 'kvm' in cat:
                        context_categories.add('kvm')
                    elif 'switch' in cat:
                        context_categories.add('switches')
                    elif 'mount' in cat:
                        context_categories.add('mounts')
                    elif 'enclosure' in cat:
                        context_categories.add('enclosures')
                    elif 'splitter' in cat:
                        context_categories.add('splitters')

        if not context_categories:
            return False  # No category info in context

        # If query has categories NOT in context, it's a different search
        new_categories = query_categories - context_categories
        return len(new_categories) > 0

    def _is_constraint_refinement(self, text: str) -> bool:
        """
        Check if this is a constraint refinement (e.g., "I need 6ft", "shorter please").
        """
        refinement_patterns = [
            # Length refinements with specific values
            r'\b(?:need|want|prefer)\s+(?:a\s+|one\s+)?\d+\s*(?:ft|foot|feet|m|meter)\b',
            r'\b\d+\s*(?:ft|foot|feet)\s*(?:version|option|one|instead|please)?\b',
            # Relative length refinements (shorter/longer)
            r'\b(?:shorter|longer)\s*(?:one|cable|version|option|please)?\b',
            r'\b(?:do you have|got|have)\s+(?:a\s+)?(?:shorter|longer)\b',
            r'\b(?:something|anything)\s+(?:shorter|longer)\b',
            # "instead" pattern (e.g., "10ft instead", "a shorter one instead")
            r'\b(?:instead|rather)\b',
            # Feature refinements
            r'\b(?:need|want)\s+(?:4k|8k|charging|power)\b',
            # Color refinements
            r'\b(?:in\s+)?(?:black|white|gray|grey)\s*(?:please|version)?\b',
        ]
        return any(re.search(pat, text) for pat in refinement_patterns)

    def _has_domain_tokens(self, text: str) -> bool:
        """Check if text contains product domain keywords."""
        domain_tokens = {
            "cable", "cables", "adapter", "adapters", "dock", "docking",
            "hub", "hubs", "kvm", "switch", "enclosure",
            "station", "stations",
            "hdmi", "displayport", "usb", "thunderbolt", "tb3", "tb4", "vga", "dvi",
            "mount", "mounts",
            "splitter", "splitters",
            "multiport",
        }

        words = set(re.findall(r'[a-z]+', text.lower()))
        return bool(words & domain_tokens)

    def _extract_sku(self, text: str) -> str | None:
        """
        Extract a product SKU from the query if present.

        StarTech SKUs are typically:
        - 5-20 characters
        - Alphanumeric with possible hyphens
        - MUST contain at least one digit (this distinguishes them from words)
        - Examples: HDMM10, CDP2HD1MBNL, DK30A2DHU, 45PATCH10BK

        Returns the SKU if found, None otherwise.
        """
        # Clean the text
        text = text.strip()

        # If the entire query is a single SKU-like token (must have at least one digit)
        if re.match(r'^[A-Za-z0-9\-]{4,25}$', text) and re.search(r'\d', text):
            return text.upper()

        # Look for SKU patterns in longer text
        # Pattern: must contain both letters and numbers, 5-20 total chars
        # This catches patterns like HDMM10, CDP2HD1MBNL, 45PATCH10BK
        words = text.split()
        for word in words:
            word_clean = word.strip('.,!?').upper()
            # Must be 5-20 chars, alphanumeric with optional hyphens
            if re.match(r'^[A-Z0-9\-]{5,20}$', word_clean):
                # Must have at least one letter AND one digit
                if re.search(r'[A-Z]', word_clean) and re.search(r'\d', word_clean):
                    return word_clean

        return None
