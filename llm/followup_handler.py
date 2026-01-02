"""
Follow-up Question Handler - Handle product context questions intelligently

Handles various types of follow-up questions about products in context:
- Single product details: "Tell me more about product 2"
- Product comparisons: "What's the difference between product 1 and 2?"
- Superlative questions: "Which one is longest?"
- General product questions: "Do any of these support 4K?"
"""

import re
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import ConversationContext

from core.context import Product, Intent


class FollowupHandler:
    """
    Handles follow-up questions about products in context.

    Provides intelligent answers to:
    - Specific product requests by number/ordinal
    - Comparison questions between products
    - Superlative questions (longest, cheapest, etc.)
    - Feature queries across products
    """

    def handle_followup(
        self,
        query: str,
        products: List[Product],
        intent: Intent,
        context: Optional["ConversationContext"] = None,
    ) -> Optional[str]:
        """
        Handle a follow-up question about products in context.

        Args:
            query: User's question
            products: Products currently in context
            intent: Classified intent with metadata
            context: Optional conversation context (for tracking comparisons)

        Returns:
            Response string, or None if can't handle
        """
        if not products:
            return None

        query_lower = query.lower()

        # PRIORITY 0: Check for comparison FOLLOW-UP questions
        # After comparing #1 and #2, "which one is better for 4K?" should stay on those two
        if context and context.has_comparison_context():
            if self._is_comparison_followup(query_lower):
                compared_products = context.get_compared_products()
                if compared_products:
                    # Handle the follow-up using only the compared products
                    followup_response = self._handle_comparison_followup(
                        query_lower, compared_products, context.last_comparison_indices, context
                    )
                    if followup_response:
                        return followup_response

        # PRIORITY 1: Check for feature questions about a SPECIFIC product
        # "Does the first one have 4K?", "Can the second product support 8K?"
        # This must come BEFORE general yes/no so we can target a single product
        specific_feature_response = self._handle_specific_product_feature(query_lower, products)
        if specific_feature_response:
            return specific_feature_response

        # PRIORITY 2: Check for yes/no questions about ALL product attributes
        # "Are they red?", "Do any of these support 4K?", "Are they both black?"
        yes_no_response = self._handle_yes_no_question(query_lower, products)
        if yes_no_response:
            return yes_no_response

        # PRIORITY 2.5: Check for feature recommendation questions
        # "Which one is better for 4K?", "Which is best for gaming?"
        # This is DIFFERENT from comparison followup - no prior comparison needed
        feature_rec_response = self._handle_feature_recommendation(query_lower, products, context)
        if feature_rec_response:
            return feature_rec_response

        # PRIORITY 3: Check for comparison question FIRST
        # "What's the difference between product 1 and 2" should be comparison, not single
        if self._is_comparison_question(query_lower):
            # Check if user wants to compare ALL products
            if self._wants_all_product_comparison(query_lower, len(products)):
                return self._handle_all_product_comparison(products)

            indices = self._extract_comparison_indices(query_lower)
            if indices:
                # Track comparison context for follow-up questions
                if context:
                    context.set_comparison_context(indices)
                return self._handle_comparison(products, indices, query_lower)

        # Clear comparison context if this is NOT a comparison or comparison follow-up
        # (user moved on to a different type of question)
        if context and context.has_comparison_context():
            context.clear_comparison_context()

        # PRIORITY 4: Check for superlative question
        superlative = self._detect_superlative(query_lower)
        if superlative:
            return self._handle_superlative(products, superlative)

        # PRIORITY 5: Check for specific product request
        product_index = self._get_product_index(query_lower, intent)
        if product_index is not None:
            return self._handle_specific_product(products, product_index, query_lower)

        # Can't handle - return None to fall back to default behavior
        return None

    def _is_comparison_followup(self, query: str) -> bool:
        """
        Check if this is a follow-up question to a previous comparison.

        Patterns like:
        - "which one is better for 4K?"
        - "which is longer?"
        - "which should I get?"
        - "what about for gaming?"
        - "and for 8K?"
        """
        patterns = [
            r'\bwhich\s+(?:one|is)\b',  # "which one", "which is"
            r'\bwhich\s+should\b',  # "which should I get"
            r'\bwhat\s+about\b',  # "what about for gaming?"
            r'^and\s+(?:for|what|which)\b',  # "and for 8K?", "and what about..."
            r'\bbetter\s+for\b',  # "better for 4K"
            r'\bbest\s+for\b',  # "best for gaming"
            r'\bprefer\b',  # "which do you prefer"
            r'\brecommend\b',  # "which do you recommend"
        ]
        return any(re.search(p, query) for p in patterns)

    def _handle_comparison_followup(
        self,
        query: str,
        compared_products: List[Product],
        indices: List[int],
        context: Optional["ConversationContext"] = None
    ) -> Optional[str]:
        """
        Handle a follow-up question about previously compared products.

        Args:
            query: User's question
            compared_products: The 2 products that were compared
            indices: Original indices (1-based) of the compared products
            context: Optional conversation context

        Returns:
            Response string, or None if can't handle
        """
        if len(compared_products) < 2:
            return None

        prod1, prod2 = compared_products[0], compared_products[1]
        sku1, sku2 = prod1.product_number, prod2.product_number

        # Check what feature they're asking about
        feature_query = self._detect_feature_query(query)

        if feature_query:
            # Compare the two products on this specific feature
            return self._compare_on_feature(prod1, prod2, feature_query, indices, context)

        # Generic "which is better" - give a balanced answer
        if re.search(r'\b(?:better|best|prefer|recommend)\b', query):
            return self._give_recommendation(prod1, prod2, indices)

        return None

    def _handle_feature_recommendation(
        self,
        query: str,
        products: List[Product],
        context: Optional["ConversationContext"] = None
    ) -> Optional[str]:
        """
        Handle "which is better/best for [feature]" questions about all products.

        This is for cases where user asks about a feature WITHOUT a prior comparison.
        Examples:
        - "Which one is better for 4K?"
        - "Which is best for gaming?"
        - "Which should I get for HDR?"

        Returns:
            Response string, or None if not a feature recommendation question
        """
        # Check if this is a "which is better/best for X" pattern
        feature_rec_patterns = [
            r'\bwhich\s+(?:one\s+)?(?:is\s+)?(?:better|best)\s+(?:for|with)\b',
            r'\bwhich\s+(?:should\s+i|do\s+you)\s+(?:get|recommend|suggest)\s+(?:for|with)\b',
            r'\bbest\s+(?:one\s+)?(?:for|with)\b',
            r'\bbetter\s+(?:for|with)\b',
            r'\brecommend\s+(?:for|with)\b',
        ]

        is_feature_rec_question = any(re.search(p, query) for p in feature_rec_patterns)
        if not is_feature_rec_question:
            return None

        # Detect which feature they're asking about
        feature = self._detect_feature_query(query)
        if not feature:
            return None

        # Special case: length - handled by superlative handler
        if feature == 'length':
            return None  # Let superlative handler deal with "longer/longest"

        # Check which products have this feature
        products_with_feature = []
        products_without_feature = []

        for i, prod in enumerate(products, 1):
            has_feature = self._product_has_feature(prod, feature)
            if has_feature:
                products_with_feature.append((i, prod))
            else:
                products_without_feature.append((i, prod))

        # Generate response based on findings
        if len(products_with_feature) == len(products):
            # ALL products have the feature
            return self._format_all_have_feature(products, feature)
        elif len(products_with_feature) > 0:
            # SOME products have the feature - recommend them
            return self._format_some_have_feature(products_with_feature, products_without_feature, feature)
        else:
            # NONE have the feature - offer to search
            return self._format_none_have_feature(products, feature, context)

    def _product_has_feature(self, product: Product, feature: str) -> bool:
        """Check if a product has a specific feature."""
        features = product.metadata.get('features', [])
        content = product.content.lower() if product.content else ''
        name = product.metadata.get('name', '').lower()
        feature_lower = feature.lower()

        # Use unified resolution checking for resolution features
        # This ensures consistent 4K/8K/1440p/1080p detection across all code paths
        if feature_lower in ('4k', '8k', '1440p', '1080p', '2k', 'qhd'):
            # Map variations to standard names
            resolution_map = {'2k': '1440p', 'qhd': '1440p'}
            resolution = resolution_map.get(feature_lower, feature_lower)
            if product.supports_resolution(resolution):
                return True
            # Also check inherent capabilities for cables
            if self._get_inherent_capability(product, feature):
                return True
            return False

        # Special handling for Power Delivery / charging
        if feature_lower in ('power delivery', 'charging', 'pd'):
            # Check dedicated metadata fields
            pd_value = product.metadata.get('power_delivery') or product.metadata.get('hub_power_delivery')
            if pd_value and str(pd_value).strip() and str(pd_value).lower() not in ('no', 'nan', ''):
                return True
            # Check features list for power delivery or charging
            if any('power delivery' in f.lower() or 'charging' in f.lower() for f in features):
                return True
            # Check content for charging with wattage
            if 'power delivery' in content or ('charging' in content and 'w' in content):
                return True
            return False

        # Check features list
        for f in features:
            if feature_lower in f.lower():
                return True

        # Check product content and name
        if feature_lower in content or feature_lower in name:
            return True

        # Special handling for gaming (usually means high refresh rate support)
        if feature == 'gaming':
            gaming_indicators = ['120hz', '144hz', '240hz', 'high refresh', 'gaming']
            return any(ind in content or ind in name for ind in gaming_indicators)

        # Check inherent capabilities based on cable type
        if self._get_inherent_capability(product, feature):
            return True

        return False

    def _format_all_have_feature(self, products: List[Product], feature: str) -> str:
        """Format response when all products have the feature."""
        if len(products) == 1:
            sku = products[0].product_number
            return f"**Yes**, the **{sku}** supports {feature}!"

        lines = [f"**Great news!** All {len(products)} products support {feature}:"]
        lines.append("")
        for i, prod in enumerate(products, 1):
            sku = prod.product_number
            length = prod.metadata.get('length_display', '')
            length_info = f" ({length})" if length else ""
            lines.append(f"- **#{i} {sku}**{length_info}")

        lines.append("")
        lines.append("Pick whichever length or price works best for you!")
        return "\n".join(lines)

    def _format_some_have_feature(
        self,
        with_feature: List[Tuple[int, Product]],
        without_feature: List[Tuple[int, Product]],
        feature: str
    ) -> str:
        """Format response when some products have the feature."""
        lines = []

        if len(with_feature) == 1:
            idx, prod = with_feature[0]
            sku = prod.product_number
            lines.append(f"For {feature}, go with **#{idx} {sku}** - it's the only one that supports it.")
        else:
            lines.append(f"For {feature}, these are your options:")
            lines.append("")
            for idx, prod in with_feature:
                sku = prod.product_number
                length = prod.metadata.get('length_display', '')
                features = prod.metadata.get('features', [])
                # Find the specific feature variant (e.g., "4K@60Hz")
                feature_detail = next((f for f in features if feature.lower() in f.lower()), feature)
                length_info = f" - {length}" if length else ""
                lines.append(f"- **#{idx} {sku}**{length_info} ({feature_detail})")

        # Mention what doesn't have it
        lines.append("")
        without_skus = [f"#{idx}" for idx, _ in without_feature]
        if len(without_feature) == 1:
            lines.append(f"Note: {without_skus[0]} does not list {feature} support.")
        else:
            lines.append(f"Note: {', '.join(without_skus)} do not list {feature} support.")

        return "\n".join(lines)

    def _format_none_have_feature(
        self,
        products: List[Product],
        feature: str,
        context: Optional["ConversationContext"] = None
    ) -> str:
        """Format response when no products have the feature - offer alternative search."""
        # Get product type from first product
        product_type = self._get_product_type(products[0])

        lines = []
        lines.append(f"**None of these** explicitly list {feature} support in their specs.")
        lines.append("")

        # Offer to find products with the feature
        offer = self._get_feature_search_offer(feature, product_type)
        lines.append(offer)

        # Track this offer so we can follow through when user says "yes"
        if context and products:
            prod = products[0]
            connector_from = prod.metadata.get('connector_from')
            connector_to = prod.metadata.get('connector_to')
            category = prod.metadata.get('CATEGORY') or prod.metadata.get('category')

            # Also check connectors list if connector_from/to not available
            if not connector_from:
                connectors = prod.metadata.get('connectors', [])
                if connectors:
                    connector_from = connectors[0] if isinstance(connectors[0], str) else str(connectors[0])
                    if len(connectors) > 1:
                        connector_to = connectors[1] if isinstance(connectors[1], str) else str(connectors[1])

            context.set_pending_feature_search(
                feature=feature,
                product_type=product_type,
                connector_from=connector_from,
                connector_to=connector_to,
                category=category
            )

        return "\n".join(lines)

    def _detect_feature_query(self, query: str) -> Optional[str]:
        """Detect what feature the user is asking about."""
        feature_patterns = {
            '4K': [r'\b4k\b'],
            '8K': [r'\b8k\b'],
            '1080p': [r'\b(?:1080p?|full\s*hd)\b'],
            '1440p': [r'\b(?:1440p?|2k|qhd)\b'],
            'HDR': [r'\bhdr(?:10)?\b'],
            'length': [r'\b(?:long|longer|length|reach)\b'],
            'gaming': [r'\bgaming\b'],
            'refresh_rate': [r'\b(?:60hz|120hz|144hz|refresh)\b'],
            'USB 3.0': [r'\busb\s*3\.?0\b'],
            'USB 3.1': [r'\busb\s*3\.?1\b'],
            'USB 3.2': [r'\busb\s*3\.?2\b'],
            'USB 2.0': [r'\busb\s*2\.?0\b'],
            'HDCP': [r'\bhdcp\b'],
            # eARC must come before ARC (earc contains arc)
            'eARC': [r'\b(?:enhanced\s+audio\s+return|earc|e-arc)\b'],
            'ARC': [r'\b(?:audio\s+return\s+channel|arc)\b'],
            'Power Delivery': [r'\b(?:power\s*delivery|pd|charg(?:ing|e))\b'],
        }
        for feature, patterns in feature_patterns.items():
            if any(re.search(p, query) for p in patterns):
                return feature
        return None

    def _compare_on_feature(
        self,
        prod1: Product,
        prod2: Product,
        feature: str,
        indices: List[int],
        context: Optional["ConversationContext"] = None
    ) -> str:
        """Compare two products on a specific feature."""
        sku1, sku2 = prod1.product_number, prod2.product_number
        feat1 = set(prod1.metadata.get('features', []))
        feat2 = set(prod2.metadata.get('features', []))

        # Special handling for length
        if feature == 'length':
            len1 = prod1.metadata.get('length_ft', 0)
            len2 = prod2.metadata.get('length_ft', 0)
            len1_display = prod1.metadata.get('length_display', '')
            len2_display = prod2.metadata.get('length_display', '')

            if len1 > len2:
                return f"**{sku1}** is longer at {len1_display} vs {len2_display} for **{sku2}**."
            elif len2 > len1:
                return f"**{sku2}** is longer at {len2_display} vs {len1_display} for **{sku1}**."
            else:
                return f"They're the same length: {len1_display}."

        # Check for feature in both products
        has_feat1 = feature in feat1 or feature.upper() in feat1
        has_feat2 = feature in feat2 or feature.upper() in feat2

        if has_feat1 and has_feat2:
            return f"Both **{sku1}** and **{sku2}** support {feature} - either will work great!"
        elif has_feat1:
            return f"**{sku1}** supports {feature}, while **{sku2}** doesn't. Go with **{sku1}** for {feature}."
        elif has_feat2:
            return f"**{sku2}** supports {feature}, while **{sku1}** doesn't. Go with **{sku2}** for {feature}."
        else:
            # Neither has the feature - offer to find products that do
            product_type = self._get_product_type(prod1)
            offer = self._get_feature_search_offer(feature, product_type)

            # Track this offer so we can follow through when user says "yes"
            # Capture connector and category info to maintain search context
            if context:
                # Extract connector info from the product
                connector_from = prod1.metadata.get('connector_from')
                connector_to = prod1.metadata.get('connector_to')
                category = prod1.metadata.get('CATEGORY') or prod1.metadata.get('category')

                # Also check connectors list if connector_from/to not available
                if not connector_from:
                    connectors = prod1.metadata.get('connectors', [])
                    if connectors:
                        connector_from = connectors[0] if isinstance(connectors[0], str) else str(connectors[0])
                        if len(connectors) > 1:
                            connector_to = connectors[1] if isinstance(connectors[1], str) else str(connectors[1])

                context.set_pending_feature_search(
                    feature=feature,
                    product_type=product_type,
                    connector_from=connector_from,
                    connector_to=connector_to,
                    category=category
                )

            return (
                f"Neither **{sku1}** nor **{sku2}** explicitly lists {feature} support in their specs.\n\n"
                f"{offer}"
            )

    def _get_product_type(self, product: Product) -> str:
        """Get a user-friendly product type description."""
        category = product.metadata.get('CATEGORY', '').lower()
        subcategory = product.metadata.get('SUBCATEGORY', '').lower()

        # Try to get connector info for cables
        connectors = product.metadata.get('connectors', [])
        if connectors and len(connectors) >= 2:
            # e.g., "HDMI cables"
            conn = connectors[0] if isinstance(connectors[0], str) else str(connectors[0])
            # Simplify connector name
            conn_simple = conn.split('(')[0].strip()
            if 'cable' in category or 'cable' in subcategory:
                return f"{conn_simple} cables"

        if 'cable' in category:
            return 'cables'
        elif 'adapter' in category:
            return 'adapters'
        elif 'dock' in category:
            return 'docks'
        elif 'hub' in category:
            return 'hubs'
        elif 'kvm' in category:
            return 'KVM switches'
        else:
            return 'products'

    def _get_feature_search_offer(self, feature: str, product_type: str) -> str:
        """Generate an offer to search for products with the missing feature."""
        # Feature-specific helpful context
        feature_context = {
            '4K': "Would you like me to find {product_type} with **4K support**? I can show you options certified for 4K @ 60Hz.",
            '8K': "Would you like me to find {product_type} with **8K support**? These require Ultra High Speed HDMI.",
            'HDR': "Would you like me to find {product_type} with **HDR support**? Great for vivid colors and contrast.",
            'Thunderbolt': "Would you like me to find **Thunderbolt** {product_type}? They offer faster speeds and daisy-chaining.",
            'Power Delivery': "Would you like me to find {product_type} with **Power Delivery**? These can charge your laptop while connected.",
            'ethernet': "Would you like me to find {product_type} with **ethernet**? Useful for a stable wired connection.",
            'gaming': "Would you like me to find {product_type} optimized for **gaming**? Look for high refresh rate support.",
            'refresh_rate': "Would you like me to find {product_type} with **high refresh rate** support (120Hz+)?",
        }

        template = feature_context.get(
            feature,
            f"Would you like me to find {product_type} with **{feature}** support?"
        )

        return template.format(product_type=product_type)

    def _give_recommendation(
        self,
        prod1: Product,
        prod2: Product,
        indices: List[int]
    ) -> str:
        """Give a balanced recommendation between two products."""
        sku1, sku2 = prod1.product_number, prod2.product_number

        # Check for feature differences
        feat1 = set(prod1.metadata.get('features', []))
        feat2 = set(prod2.metadata.get('features', []))
        only_in_1 = feat1 - feat2
        only_in_2 = feat2 - feat1

        # Check length difference
        len1 = prod1.metadata.get('length_ft', 0)
        len2 = prod2.metadata.get('length_ft', 0)

        lines = []

        if only_in_1 or only_in_2:
            lines.append("It depends on what features you need:")
            if only_in_1:
                lines.append(f"- **{sku1}** if you need: {', '.join(only_in_1)}")
            if only_in_2:
                lines.append(f"- **{sku2}** if you need: {', '.join(only_in_2)}")
        elif len1 != len2:
            if len1 > len2:
                longer, shorter = sku1, sku2
            else:
                longer, shorter = sku2, sku1
            lines.append(f"They have the same features, so it comes down to length:")
            lines.append(f"- **{longer}** for more reach")
            lines.append(f"- **{shorter}** for a tidier setup")
        else:
            lines.append(f"**{sku1}** and **{sku2}** are essentially identical - pick whichever is in stock or cheaper!")

        return "\n".join(lines)

    def _get_product_index(self, query: str, intent: Intent) -> Optional[int]:
        """Get product index from intent metadata or query."""
        # Try intent metadata first
        if intent.meta_info and 'product_index' in intent.meta_info:
            return intent.meta_info['product_index']

        # Try extracting from query
        return self._extract_product_index(query)

    def _extract_product_index(self, text: str) -> Optional[int]:
        """Extract product index from text."""
        # Direct number patterns
        match = re.search(r'\bproduct\s*(\d)\b', text)
        if match:
            return int(match.group(1))

        match = re.search(r'\bnumber\s*(\d)\b', text)
        if match:
            return int(match.group(1))

        match = re.search(r'#(\d)\b', text)
        if match:
            return int(match.group(1))

        # Ordinal words
        ordinal_map = {
            'first': 1, '1st': 1,
            'second': 2, '2nd': 2,
            'third': 3, '3rd': 3,
            'middle': 2,
            'last': 3,
        }

        for word, index in ordinal_map.items():
            if re.search(rf'\b{word}\b', text):
                return index

        return None

    # Feature patterns for specific product questions
    # "Does the first one have 4K?", "Can product 2 do HDR?"
    SPECIFIC_PRODUCT_FEATURES = {
        '4k': {
            'patterns': [r'\b4k\b'],
            'field': 'features',
            'value': '4K',
            'display': '4K',
        },
        '8k': {
            'patterns': [r'\b8k\b'],
            'field': 'features',
            'value': '8K',
            'display': '8K',
        },
        'hdr': {
            'patterns': [r'\bhdr\b'],
            'field': 'features',
            'value': 'HDR',
            'display': 'HDR',
        },
        'arc': {
            'patterns': [r'\barc\b', r'\bearc\b'],
            'field': 'features',
            'value': 'ARC',
            'display': 'ARC (Audio Return Channel)',
        },
        'thunderbolt': {
            'patterns': [r'\bthunderbolt\b'],
            'field': 'features',
            'value': 'Thunderbolt',
            'display': 'Thunderbolt',
        },
        'power_delivery': {
            'patterns': [r'\bpower\s*delivery\b', r'\bpd\b', r'\busb-?pd\b', r'\bcharging?\b', r'\bcharge\b'],
            'field': 'features',
            'value': 'Power Delivery',
            'display': 'Power Delivery',
        },
        'ethernet': {
            'patterns': [r'\bethernet\b'],
            'field': 'features',
            'value': 'Ethernet',
            'display': 'Ethernet',
        },
    }

    def _handle_specific_product_feature(
        self,
        query: str,
        products: List[Product]
    ) -> Optional[str]:
        """
        Handle feature questions about a specific product by ordinal reference.

        Examples:
        - "Does the first one have 4K support?"
        - "Can product 2 do HDR?"
        - "Is the second one Thunderbolt compatible?"

        Returns:
            Answer string, or None if not a specific product feature question
        """
        # First, detect if query references a specific product by ordinal
        product_index = self._extract_product_index(query)
        if product_index is None:
            return None

        # Validate index
        if product_index < 1 or product_index > len(products):
            return None

        # Check if query asks about a feature
        asked_feature = None
        for feature_key, config in self.SPECIFIC_PRODUCT_FEATURES.items():
            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    asked_feature = config
                    break
            if asked_feature:
                break

        if not asked_feature:
            return None  # Not a feature question, let other handlers try

        # Get the specific product
        product = products[product_index - 1]
        sku = product.product_number
        name = product.metadata.get('name', sku)
        feature_display = asked_feature['display']
        feature_value = asked_feature['value']
        field = asked_feature['field']

        # Check if product has the feature
        has_feature = False

        # Special handling for Power Delivery - check multiple metadata fields
        if feature_value.lower() == 'power delivery':
            pd_value = product.metadata.get('power_delivery') or product.metadata.get('hub_power_delivery')
            if pd_value and str(pd_value).strip() and str(pd_value).lower() not in ('no', 'nan', ''):
                has_feature = True
            if not has_feature:
                features = product.metadata.get('features', [])
                has_feature = any('power delivery' in f.lower() or 'charging' in f.lower() for f in features)
            if not has_feature:
                content = product.content.lower() if product.content else ''
                has_feature = 'power delivery' in content or ('charging' in content and 'w' in content)
        elif field == 'features':
            features = product.metadata.get('features', [])
            # Check both exact match and substring match (for "4K" in "4K@60Hz")
            has_feature = any(
                feature_value.lower() in f.lower()
                for f in features
            )

            # Also check the product description/content for feature mentions
            if not has_feature:
                content = product.content.lower() if product.content else ''
                product_name = name.lower() if name else ''
                has_feature = (
                    feature_value.lower() in content or
                    feature_value.lower() in product_name
                )
        else:
            field_value = product.metadata.get(field, '')
            has_feature = feature_value.lower() in str(field_value).lower()

        # Build response
        if has_feature:
            response = f"**Yes**, the **{sku}** supports {feature_display}."

            # Add extra context if available
            features = product.metadata.get('features', [])
            related_features = [f for f in features if feature_value.lower() in f.lower()]
            if related_features:
                response += f" (Specifically: {', '.join(related_features)})"
        else:
            response = f"**No**, the **{sku}** does not appear to support {feature_display}."

            # Suggest alternatives if other products have this feature
            alternatives = []
            for i, prod in enumerate(products, 1):
                if i == product_index:
                    continue
                prod_features = prod.metadata.get('features', [])
                if any(feature_value.lower() in f.lower() for f in prod_features):
                    alternatives.append(f"Product {i} ({prod.product_number})")

            if alternatives:
                response += f"\n\nHowever, {' and '.join(alternatives)} {'does' if len(alternatives) == 1 else 'do'} support {feature_display}."

        return response

    # Map common question keywords to metadata field names
    QUESTION_FIELD_MAP = {
        # Temperature related
        'operating temp': 'operating_temp',
        'operating temperature': 'operating_temp',
        'temperature range': 'operating_temp',
        'temp range': 'operating_temp',
        'storage temp': 'STORAGETEMP',
        'storage temperature': 'STORAGETEMP',
        # Environmental
        'humidity': 'humidity',
        # Power related
        'power consumption': 'POWERCONSUMPTION',
        'power': 'power_adapter',
        'wattage': 'POWERCONSUMPTION',
        'voltage': 'INPUTVOLTS',
        # Physical dimensions
        'dimension': 'product_length',
        'dimensions': 'product_length',
        'size': 'product_length',
        'weight': 'product_weight',
        'height': 'product_height',
        'length': 'length_display',
        'width': 'product_width',
        # Networking
        'speed': 'network_speed',
        'bandwidth': 'MAXDATARATE',
        'max distance': 'MAXDISTANCE',
        'mtbf': 'MTBF',
        # Materials
        'material': 'AMZ_MAT',
        'enclosure': 'ENCLOSURETYPE',
        # Other common specs
        'warranty': 'warranty',
        'color': 'color',
        'port': 'hub_ports',
        'ports': 'hub_ports',
    }

    def _detect_specific_question(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Detect if user is asking about a specific product attribute.

        Returns:
            Tuple of (display_name, field_name) if detected, None otherwise
        """
        query_lower = query.lower()

        for keyword, field in self.QUESTION_FIELD_MAP.items():
            if keyword in query_lower:
                # Generate display name from keyword
                display_name = keyword.title()
                return (display_name, field)

        return None

    def _clean_html_entities(self, value: str) -> str:
        """Clean HTML entities from value strings."""
        if not isinstance(value, str):
            return str(value)

        # Common HTML entity replacements
        value = value.replace('&amp;deg;', '°')
        value = value.replace('&deg;', '°')
        value = value.replace('&lt;', '<')
        value = value.replace('&gt;', '>')
        value = value.replace('&amp;', '&')
        value = value.replace('<br>', ', ')
        value = value.replace('</br>', '')
        value = value.replace('<br/>', ', ')
        value = value.replace('_x000D_', '')

        return value.strip()

    def _handle_specific_product(
        self,
        products: List[Product],
        index: int,
        query: str
    ) -> str:
        """Handle request for specific product details."""
        # Validate index
        if index < 1 or index > len(products):
            return f"I only showed you {len(products)} products. Please specify product 1-{len(products)}."

        product = products[index - 1]
        name = product.metadata.get('name', product.product_number)
        sku = product.product_number

        lines = []

        # Check if user is asking about a specific field
        specific_question = self._detect_specific_question(query)

        if specific_question:
            display_name, field_name = specific_question
            value = product.metadata.get(field_name)

            if value:
                # Clean up HTML entities
                clean_value = self._clean_html_entities(str(value))
                # Answer the specific question FIRST
                lines.append(f"The **{sku}** has an {display_name.lower()} of **{clean_value}**.")
                lines.append("")
                lines.append("Here are the full specs:")
            else:
                # Field not available for this product
                lines.append(f"The {display_name.lower()} specification is not available for **{sku}**.")
                lines.append("")
                lines.append("Here's what I do have:")
        else:
            # Generic "tell me more" request
            lines.append(f"Here's more about **{name}** ({sku}):")

        lines.append("")

        # Category
        category = product.metadata.get('category', '')
        if category:
            lines.append(f"**Category:** {category.replace('_', ' ').title()}")

        # Length (for cables)
        length_display = product.metadata.get('length_display')
        if length_display:
            lines.append(f"**Length:** {length_display}")

        # Connectors
        connectors = product.metadata.get('connectors', [])
        if connectors and len(connectors) >= 2:
            lines.append(f"**Connectors:** {connectors[0]} to {connectors[1]}")

        # Features
        features = product.metadata.get('features', [])
        if features:
            lines.append(f"**Features:** {', '.join(features)}")

        # Network-specific fields
        network_speed = product.metadata.get('network_speed')
        if network_speed:
            lines.append(f"**Speed:** {network_speed}")

        hub_ports = product.metadata.get('hub_ports')
        if hub_ports:
            lines.append(f"**Ports:** {hub_ports}")

        # Environmental specs (important for industrial/outdoor)
        operating_temp = product.metadata.get('operating_temp')
        if operating_temp:
            lines.append(f"**Operating Temperature:** {self._clean_html_entities(operating_temp)}")

        humidity = product.metadata.get('humidity')
        if humidity:
            lines.append(f"**Humidity:** {self._clean_html_entities(humidity)}")

        storage_temp = product.metadata.get('STORAGETEMP')
        if storage_temp:
            lines.append(f"**Storage Temperature:** {self._clean_html_entities(storage_temp)}")

        # Power specs
        power_consumption = product.metadata.get('POWERCONSUMPTION')
        if power_consumption:
            lines.append(f"**Power Consumption:** {self._clean_html_entities(power_consumption)}")

        power_adapter = product.metadata.get('power_adapter')
        if power_adapter:
            lines.append(f"**Power:** {power_adapter}")

        # Physical specs
        wire_gauge = product.metadata.get('wire_gauge')
        if wire_gauge:
            lines.append(f"**Wire Gauge:** {wire_gauge}")

        connector_plating = product.metadata.get('connector_plating')
        if connector_plating:
            lines.append(f"**Plating:** {connector_plating}")

        shield_type = product.metadata.get('shield_type')
        if shield_type:
            lines.append(f"**Shielding:** {shield_type}")

        material = product.metadata.get('AMZ_MAT')
        if material:
            lines.append(f"**Material:** {material}")

        # Color and warranty
        color = product.metadata.get('color')
        if color:
            lines.append(f"**Color:** {color}")

        warranty = product.metadata.get('warranty')
        if warranty:
            lines.append(f"**Warranty:** {warranty}")

        lines.append("")
        lines.append("Would you like to compare this with the other products, or ask about specific features?")

        return "\n".join(lines)

    def _is_comparison_question(self, query: str) -> bool:
        """Check if query is asking for comparison."""
        comparison_patterns = [
            r'\b(?:what\'?s|what\s+is)\s+(?:the\s+)?difference\b',
            r'\bcompare\b',
            r'\bdifferent\b',
            r'\bvs\.?\b',
            r'\bversus\b',
            r'\bproduct\s*\d\s+(?:and|&|vs\.?|versus)\s+(?:product\s*)?\d\b',
            r'\b(?:between|comparing)\b',
            # "Which is better?" style questions trigger comparison
            r'\bwhich\s+(?:one\s+)?(?:is\s+)?(?:better|best)\b',
            r'\bwhat\s+(?:do\s+you|would\s+you)\s+recommend\b',
            r'\bwhich\s+(?:should\s+i|would\s+you)\s+(?:get|choose|pick|recommend)\b',
        ]
        return any(re.search(pat, query) for pat in comparison_patterns)

    def _extract_comparison_indices(self, query: str) -> List[int]:
        """Extract product indices for comparison."""
        # "#1 and #2" format
        match = re.search(r'#(\d)\s+(?:and|&|vs\.?|versus)\s+#?(\d)', query)
        if match:
            return [int(match.group(1)), int(match.group(2))]

        # "product 1 and 2", "products 1 and 2", "between 1 and 2"
        match = re.search(r'(?:product\s*)?(\d)\s+(?:and|&|vs\.?|versus)\s+(?:product\s*)?(\d)', query)
        if match:
            return [int(match.group(1)), int(match.group(2))]

        # "between product 1 and product 2"
        match = re.search(r'between\s+(?:product\s*)?(\d)\s+and\s+(?:product\s*)?(\d)', query)
        if match:
            return [int(match.group(1)), int(match.group(2))]

        # "between #1 and #2"
        match = re.search(r'between\s+#?(\d)\s+and\s+#?(\d)', query)
        if match:
            return [int(match.group(1)), int(match.group(2))]

        # "first and second", "1st and 2nd"
        ordinals = {
            'first': 1, '1st': 1,
            'second': 2, '2nd': 2,
            'third': 3, '3rd': 3,
        }
        match = re.search(r'\b(first|1st|second|2nd|third|3rd)\s+(?:one\s+)?(?:and|&)\s+(?:the\s+)?(first|1st|second|2nd|third|3rd)', query)
        if match:
            idx1 = ordinals.get(match.group(1).lower())
            idx2 = ordinals.get(match.group(2).lower())
            if idx1 and idx2:
                return [idx1, idx2]

        return []

    def _wants_all_product_comparison(self, query: str, num_products: int) -> bool:
        """Check if user wants to compare ALL products or get a recommendation."""
        all_patterns = [
            r'\bcompare\s+(?:all|them|these|the\s+options)\b',
            r'\ball\s+(?:three|3|four|4|five|5)\b',
            r'\b(?:compare|difference)\s+(?:between\s+)?(?:all|them|these)\b',
            r'\bhow\s+(?:do\s+)?they\s+(?:all\s+)?compare\b',
            r'\bcompare\s+(?:the\s+)?(?:options|products|choices)\b',
            # "Which is better?" without specific product numbers = compare all
            r'\bwhich\s+(?:one\s+)?(?:is\s+)?(?:better|best)\b(?!\s+(?:for|with|between|#|\d))',
            r'\bwhat\s+(?:do\s+you|would\s+you)\s+recommend\b',
            r'\bwhich\s+(?:should\s+i|would\s+you)\s+(?:get|choose|pick|recommend)\b',
        ]
        return any(re.search(p, query) for p in all_patterns)

    def _handle_all_product_comparison(self, products: List[Product]) -> str:
        """
        Handle comparison of ALL products with consultative decision guidance.

        Acts like a $160k expert consultant - synthesizes differences and
        helps customer make a decision rather than just dumping specs.
        """
        if len(products) < 2:
            return "There's only one product to look at."

        # Detect product type to customize comparison
        category = products[0].metadata.get('category', '').lower()
        is_dock = category in ('dock', 'hub', 'docking_station')
        is_cable = category in ('cable', 'adapter') or products[0].metadata.get('length_ft')

        if is_dock:
            return self._consultative_dock_comparison(products)
        elif is_cable:
            return self._consultative_cable_comparison(products)
        else:
            return self._consultative_generic_comparison(products)

    def _consultative_dock_comparison(self, products: List[Product]) -> str:
        """Consultative comparison for docking stations."""
        lines = ["That depends on your priorities! Let me break it down:\n"]

        # Gather dock-specific data
        dock_data = []
        for i, prod in enumerate(products, 1):
            meta = prod.metadata
            pd_val = meta.get('power_delivery') or meta.get('hub_power_delivery')
            pd_wattage = 0
            if pd_val and str(pd_val).lower() not in ('no', 'nan', ''):
                try:
                    pd_wattage = int(str(pd_val).replace('W', '').strip())
                except:
                    pd_wattage = 1  # Has PD but unknown wattage

            dock_data.append({
                'index': i,
                'sku': prod.product_number,
                'monitors': meta.get('DOCKNUMDISPLAYS') or meta.get('dock_num_displays') or 1,
                'has_4k': prod.supports_4k(),
                'pd_wattage': pd_wattage,
                'has_ethernet': 'ethernet' in (prod.content or '').lower() or meta.get('network_speed'),
                'usb_ports': meta.get('hub_ports') or 0,
                'features': set(meta.get('features', [])),
            })

        # Find the key differentiators
        has_pd = [d for d in dock_data if d['pd_wattage'] > 0]
        no_pd = [d for d in dock_data if d['pd_wattage'] == 0]
        has_4k = [d for d in dock_data if d['has_4k']]
        no_4k = [d for d in dock_data if not d['has_4k']]

        # Build consultative response
        if has_pd and no_pd:
            # Major difference: charging capability
            pd_indices = [f"#{d['index']}" for d in has_pd]
            no_pd_indices = [f"#{d['index']}" for d in no_pd]

            lines.append("**The key difference is laptop charging:**\n")

            if len(has_pd) == 1:
                d = has_pd[0]
                wattage = f"{d['pd_wattage']}W" if d['pd_wattage'] > 1 else ""
                lines.append(f"• **#{d['index']} ({d['sku']})** can charge your laptop{' at ' + wattage if wattage else ''}")
            else:
                lines.append(f"• {', '.join(pd_indices)} can charge your laptop")

            lines.append(f"• {', '.join(no_pd_indices)} {'does' if len(no_pd) == 1 else 'do'} NOT charge - you'll need a separate charger\n")

        # 4K difference
        if has_4k and no_4k and len(has_4k) != len(dock_data):
            lines.append("**Display quality:**")
            if has_4k:
                indices_4k = ', '.join([f"#{d['index']}" for d in has_4k])
                lines.append(f"• {indices_4k} support{'s' if len(has_4k) == 1 else ''} 4K displays")
            if no_4k:
                indices_no_4k = ', '.join([f"#{d['index']}" for d in no_4k])
                lines.append(f"• {indices_no_4k} max{'es' if len(no_4k) == 1 else ''} out at 1080p")
            lines.append("")

        # Bottom line recommendations
        lines.append("**Bottom line:**")

        # Recommend based on charging
        if has_pd and no_pd:
            best_pd = max(has_pd, key=lambda x: x['pd_wattage'])
            lines.append(f"• **Want one cable for everything** (including charging)? → **#{best_pd['index']}**")
            # Find best non-PD option - prioritize 4K capability
            best_no_pd = next((d for d in no_pd if d['has_4k']), no_pd[0])
            lines.append(f"• **Need 4K resolution** and already have a charger? → **#{best_no_pd['index']}**")
        else:
            # All similar - recommend based on other factors
            if has_4k:
                lines.append(f"• All are solid choices for dual monitors")
            if dock_data:
                lines.append(f"• Pick based on your display resolution and port needs")

        # Clarifying question
        lines.append("\n**What matters most to you** - laptop charging, 4K displays, or port count?")

        return "\n".join(lines)

    def _consultative_cable_comparison(self, products: List[Product]) -> str:
        """Consultative comparison for cables/adapters."""
        lines = []

        # Gather cable data
        cable_data = []
        for i, prod in enumerate(products, 1):
            meta = prod.metadata
            cable_data.append({
                'index': i,
                'sku': prod.product_number,
                'length': meta.get('length_display', ''),
                'length_ft': meta.get('length_ft', 0),
                'connector_from': self._simplify_connector(meta.get('connectors', [''])[0] if meta.get('connectors') else ''),
                'connector_to': self._simplify_connector(meta.get('connectors', ['', ''])[1] if len(meta.get('connectors', [])) > 1 else ''),
                'features': set(meta.get('features', [])),
            })

        # Check for connector differences (critical!)
        unique_connectors = set((d['connector_from'], d['connector_to']) for d in cable_data)

        if len(unique_connectors) > 1:
            # Different connectors - must warn!
            lines.append("⚠️ **Wait!** These connect to different ports:\n")
            for d in cable_data:
                conn = f"{d['connector_from']} → {d['connector_to']}" if d['connector_to'] else d['connector_from']
                lines.append(f"• **#{d['index']}** ({d['sku']}): {conn}")
            lines.append("\nMake sure you pick the right connector for your devices!")
            return "\n".join(lines)

        # Same connectors - they're basically the same cable in different lengths
        lines.append("These are the **same cable** in different lengths:\n")

        sorted_cables = sorted(cable_data, key=lambda x: x['length_ft'] or 0)

        for d in sorted_cables:
            length = d['length'] if d['length'] else "standard length"
            lines.append(f"• **#{d['index']}** ({d['sku']}): {length}")

        # Common features
        all_features = set()
        for d in cable_data:
            all_features.update(d['features'])

        if all_features:
            lines.append(f"\nAll support: {', '.join(list(all_features)[:3])}")

        # Decision guidance
        lines.append("\n**Pick based on distance:**")
        shortest = sorted_cables[0]
        longest = sorted_cables[-1]
        lines.append(f"• Desk setup (close connection)? → **#{shortest['index']}** ({shortest['length']})")
        if len(sorted_cables) > 1:
            lines.append(f"• Need more reach? → **#{longest['index']}** ({longest['length']})")

        lines.append("\nHow far apart are your devices?")

        return "\n".join(lines)

    def _consultative_generic_comparison(self, products: List[Product]) -> str:
        """Generic consultative comparison for other product types."""
        lines = ["Here's how they compare:\n"]

        # Gather data
        product_data = []
        for i, prod in enumerate(products, 1):
            product_data.append({
                'index': i,
                'sku': prod.product_number,
                'features': set(prod.metadata.get('features', [])),
                'name': prod.metadata.get('name', prod.product_number),
            })

        # Find common and unique features
        all_features = set()
        for d in product_data:
            all_features.update(d['features'])

        common = all_features.copy()
        for d in product_data:
            common &= d['features']

        # List products with their unique features
        for d in product_data:
            unique = d['features'] - common
            unique_str = f" - unique: {', '.join(list(unique)[:2])}" if unique else ""
            lines.append(f"• **#{d['index']}** ({d['sku']}){unique_str}")

        if common:
            lines.append(f"\n**All have:** {', '.join(list(common)[:3])}")

        # Decision guidance
        lines.append("\n**Which to choose:**")
        for d in product_data:
            unique = d['features'] - common
            if unique:
                feat = list(unique)[0]
                lines.append(f"• Need **{feat}**? → **#{d['index']}**")

        lines.append("\nWhat features matter most for your use case?")

        return "\n".join(lines)

    def _simplify_connector(self, connector: str) -> str:
        """Simplify connector name for comparison."""
        if not connector:
            return ""
        # "1 x HDMI (19 pin)" -> "HDMI"
        # "1 x Mini HDMI (19 pin)" -> "Mini HDMI"
        # "1 x Micro HDMI (19 pin)" -> "Micro HDMI"
        match = re.search(r'\d+\s*x\s*(.+?)(?:\s*\(|$)', connector)
        if match:
            return match.group(1).strip()
        return connector.split('(')[0].strip()

    def _get_device_examples(self, connector_type: str) -> str:
        """Get example devices for a connector type."""
        examples = {
            'Mini HDMI': '(tablets, cameras)',
            'Micro HDMI': '(smartphones, action cameras, drones)',
            'HDMI': '(TVs, monitors, projectors)',
            'USB-C': '(modern laptops, phones)',
            'USB-A': '(computers, chargers)',
            'DisplayPort': '(monitors, graphics cards)',
            'Mini DisplayPort': '(older MacBooks, Surface)',
            'VGA': '(older monitors, projectors)',
            'DVI': '(older monitors)',
        }
        return examples.get(connector_type, '')

    def _handle_comparison(
        self,
        products: List[Product],
        indices: List[int],
        query: str
    ) -> str:
        """
        Handle comparison between specific products.

        Tone: $160K CSR - conversational, helpful, explains why differences matter.
        Structure:
        1. Use actual SKUs (not "Product 1")
        2. Mention what's the SAME first (builds context)
        3. Highlight what's DIFFERENT
        4. Help them decide based on differences
        """
        # Validate indices
        valid_indices = [i for i in indices if 1 <= i <= len(products)]
        if len(valid_indices) < 2:
            return f"I can only compare products 1-{len(products)}. Please specify valid product numbers."

        prod1 = products[valid_indices[0] - 1]
        prod2 = products[valid_indices[1] - 1]

        sku1 = prod1.product_number
        sku2 = prod2.product_number

        # Gather comparison data
        len1 = prod1.metadata.get('length_display', '')
        len2 = prod2.metadata.get('length_display', '')
        len1_ft = prod1.metadata.get('length_ft', 0)
        len2_ft = prod2.metadata.get('length_ft', 0)

        conn1 = prod1.metadata.get('connectors', [])
        conn2 = prod2.metadata.get('connectors', [])

        feat1 = set(prod1.metadata.get('features', []))
        feat2 = set(prod2.metadata.get('features', []))
        common_features = feat1 & feat2
        only_in_1 = feat1 - feat2
        only_in_2 = feat2 - feat1

        # Extended specs for similarity check
        plating1 = prod1.metadata.get('connector_plating', '')
        plating2 = prod2.metadata.get('connector_plating', '')
        shield1 = prod1.metadata.get('shield_type', '')
        shield2 = prod2.metadata.get('shield_type', '')

        # Determine what's similar vs different
        same_connectors = conn1 == conn2
        same_length = len1 == len2
        same_features = feat1 == feat2

        # Build conversational response
        lines = ["Good question! "]

        # --- SIMILARITIES FIRST ---
        similarities = []

        # Same connectors
        if same_connectors and len(conn1) >= 2:
            conn_desc = self._simplify_connector_pair(conn1)
            similarities.append(f"connect {conn_desc}")

        # Common features
        if common_features:
            feat_str = self._format_features_naturally(common_features)
            if feat_str:
                similarities.append(feat_str)

        # Same plating
        if plating1 and plating1 == plating2 and 'gold' in plating1.lower():
            similarities.append("gold-plated connectors")

        # Build similarity statement
        if similarities:
            if same_length and same_features:
                lines[0] += f"**{sku1}** and **{sku2}** are nearly identical - "
            else:
                lines[0] += f"**{sku1}** and **{sku2}** are basically the same cable - "
            lines[0] += f"both {', '.join(similarities)}."
        else:
            lines[0] += f"Let me compare **{sku1}** and **{sku2}** for you."

        lines.append("")

        # --- DIFFERENCES ---
        differences = []

        # Length difference
        if not same_length and len1 and len2:
            lines.append("**The main difference is length:**")
            # Determine which is longer and add helpful context
            if len1_ft < len2_ft:
                shorter, longer = (sku1, len1), (sku2, len2)
            else:
                shorter, longer = (sku2, len2), (sku1, len1)

            lines.append(f"- **{shorter[0]}:** {shorter[1]} - good for close connections")
            lines.append(f"- **{longer[0]}:** {longer[1]} - gives you more reach")
            differences.append('length')
            lines.append("")

        # Connector difference
        if not same_connectors:
            lines.append("**Different connectors:**")
            if len(conn1) >= 2:
                lines.append(f"- **{sku1}:** {self._simplify_connector_pair(conn1)}")
            if len(conn2) >= 2:
                lines.append(f"- **{sku2}:** {self._simplify_connector_pair(conn2)}")
            differences.append('connectors')
            lines.append("")

        # Feature differences
        if only_in_1 or only_in_2:
            lines.append("**Feature differences:**")
            if only_in_1:
                lines.append(f"- **{sku1}** has: {', '.join(only_in_1)}")
            if only_in_2:
                lines.append(f"- **{sku2}** has: {', '.join(only_in_2)}")
            differences.append('features')
            lines.append("")

        # --- RECOMMENDATION ---
        if not differences:
            lines.append("These are essentially the same product with identical specs!")
        elif differences == ['length']:
            # Only length differs - common case, give clear guidance
            lines.append("Both will perform the same, so it really comes down to how much cable length you need.")
        elif 'connectors' in differences:
            lines.append("Make sure to match the connector to what your device needs!")

        return "\n".join(lines)

    def _simplify_connector(self, connector: str) -> str:
        """Simplify connector name for display."""
        # Remove quantity prefix like "1 x"
        cleaned = re.sub(r'^\d+\s*x\s*', '', str(connector), flags=re.IGNORECASE).strip()
        cleaned_lower = cleaned.lower()

        # Map to friendly names
        if 'mini hdmi' in cleaned_lower:
            return "Mini HDMI"
        elif 'micro hdmi' in cleaned_lower:
            return "Micro HDMI"
        elif 'hdmi' in cleaned_lower:
            return "HDMI"
        elif 'usb-c' in cleaned_lower or 'type-c' in cleaned_lower:
            return "USB-C"
        elif 'mini displayport' in cleaned_lower or 'mini dp' in cleaned_lower:
            return "Mini DisplayPort"
        elif 'displayport' in cleaned_lower:
            return "DisplayPort"
        elif 'thunderbolt' in cleaned_lower:
            return "Thunderbolt"
        elif 'vga' in cleaned_lower:
            return "VGA"
        elif 'dvi' in cleaned_lower:
            return "DVI"
        else:
            # Remove parentheses content for cleaner display
            cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()
            return cleaned if cleaned else str(connector)

    def _simplify_connector_pair(self, connectors: List[str]) -> str:
        """Simplify connector pair to readable format like 'HDMI to Mini HDMI'."""
        if len(connectors) < 2:
            return "unknown connectors"

        source = self._simplify_connector(connectors[0])
        target = self._simplify_connector(connectors[1])

        return f"{source} to {target}"

    def _format_features_naturally(self, features: set) -> str:
        """Format features into natural language."""
        if not features:
            return ""

        feature_list = list(features)
        if len(feature_list) == 1:
            return f"{feature_list[0].lower()} construction"
        elif len(feature_list) == 2:
            return f"{feature_list[0].lower()} and {feature_list[1].lower()}"
        else:
            return f"{', '.join(f.lower() for f in feature_list[:-1])}, and {feature_list[-1].lower()}"

    def _detect_superlative(self, query: str) -> Optional[str]:
        """Detect superlative question type."""
        superlative_map = {
            r'\b(?:which|what)(?:\s+one)?\s+is\s+(?:the\s+)?longest\b': 'longest',
            r'\blongest\b': 'longest',
            r'\b(?:which|what)(?:\s+one)?\s+is\s+(?:the\s+)?shortest\b': 'shortest',
            r'\bshortest\b': 'shortest',
            r'\b(?:which|what)(?:\s+one)?\s+is\s+(?:the\s+)?cheapest\b': 'cheapest',
            r'\bcheapest\b': 'cheapest',
            r'\b(?:which|what)(?:\s+one)?\s+is\s+(?:the\s+)?best\b': 'best',
        }

        for pattern, superlative_type in superlative_map.items():
            if re.search(pattern, query):
                return superlative_type

        return None

    def _handle_superlative(self, products: List[Product], superlative: str) -> str:
        """Handle superlative questions (longest, shortest, etc.)."""
        if superlative in ['longest', 'shortest']:
            return self._handle_length_superlative(products, superlative)
        elif superlative == 'cheapest':
            return self._handle_price_superlative(products)
        elif superlative == 'best':
            return self._handle_best_question(products)

        return None

    def _handle_length_superlative(self, products: List[Product], superlative: str) -> str:
        """Handle longest/shortest questions."""
        # Build list of (product_index, length_ft, display)
        lengths = []
        for i, prod in enumerate(products, 1):
            length_ft = prod.metadata.get('length_ft')
            display = prod.metadata.get('length_display', 'Unknown')
            sku = prod.product_number
            name = prod.metadata.get('name', sku)

            if length_ft is not None:
                lengths.append((i, length_ft, display, sku, name))

        if not lengths:
            return "Length information isn't available for these products."

        # Sort by length
        lengths.sort(key=lambda x: x[1], reverse=(superlative == 'longest'))
        winner = lengths[0]

        idx, length_ft, display, sku, name = winner

        return f"**Product {idx}** ({sku}) is the {superlative} at **{display}**."

    def _handle_price_superlative(self, products: List[Product]) -> str:
        """Handle cheapest question."""
        # Check if we have price data
        prices = []
        for i, prod in enumerate(products, 1):
            price = prod.metadata.get('price')
            if price:
                prices.append((i, price, prod.product_number))

        if not prices:
            return "Price information isn't available in my product data. Check startech.com for current pricing."

        prices.sort(key=lambda x: x[1])
        idx, price, sku = prices[0]

        return f"**Product {idx}** ({sku}) is the cheapest at **${price:.2f}**."

    def _handle_best_question(self, products: List[Product]) -> str:
        """Handle 'which is best' - subjective, so explain trade-offs."""
        lines = ["That depends on your needs:", ""]

        # Find product with most features
        max_features = 0
        feature_leader = None
        for i, prod in enumerate(products, 1):
            feat_count = len(prod.metadata.get('features', []))
            if feat_count > max_features:
                max_features = feat_count
                feature_leader = (i, prod)

        if feature_leader:
            idx, prod = feature_leader
            features = prod.metadata.get('features', [])
            lines.append(f"**Most features:** Product {idx} ({prod.product_number}) with {', '.join(features)}")

        # Find longest cable
        lengths = [(i, prod.metadata.get('length_ft', 0), prod) for i, prod in enumerate(products, 1)]
        lengths.sort(key=lambda x: x[1], reverse=True)
        if lengths and lengths[0][1] > 0:
            idx, length, prod = lengths[0]
            display = prod.metadata.get('length_display', f'{length}ft')
            lines.append(f"**Longest reach:** Product {idx} ({prod.product_number}) at {display}")

        # Find shortest cable (if different)
        if lengths and lengths[-1][1] > 0 and lengths[-1][1] != lengths[0][1]:
            idx, length, prod = lengths[-1]
            display = prod.metadata.get('length_display', f'{length}ft')
            lines.append(f"**Most compact:** Product {idx} ({prod.product_number}) at {display}")

        lines.append("")
        lines.append("What's most important for your setup - cable length, specific features, or something else?")

        return "\n".join(lines)

    # === Yes/No Question Handling ===

    # Attribute patterns for yes/no questions
    YES_NO_ATTRIBUTE_PATTERNS = {
        # Color questions
        'color': {
            'patterns': [
                r'\b(?:are|is)\s+(?:they|it|this|these|those|the\s+cables?|the\s+products?)\s+(?:all\s+)?(\w+)\b',
                r'\b(?:are|is)\s+(?:they|it|this|these|those)\s+(?:both\s+)?(\w+)\b',
                r'\bdo\s+(?:they|this|these|any)\s+come\s+in\s+(\w+)\b',
            ],
            'field': 'color',
            'color_words': ['red', 'black', 'white', 'blue', 'gray', 'grey', 'silver', 'green', 'yellow', 'orange', 'pink', 'purple', 'gold', 'beige', 'brown'],
        },
        # 4K support
        '4k': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|this|these|those|any)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*4k\b',
                r'\b4k\s+(?:support|compatible|capable)\b',
            ],
            'field': 'features',
            'value': '4K',
        },
        # Thunderbolt
        'thunderbolt': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|this|these|those|any)\s+(?:all\s+)?(?:support|have)?\s*thunderbolt\b',
                r'\bthunderbolt\s+(?:support|compatible|capable)\b',
            ],
            'field': 'features',
            'value': 'Thunderbolt',
        },
        # Power delivery / charging
        'power_delivery': {
            'patterns': [
                r'\b(?:do|does|are|is|can|will)\s+(?:they|it|these|those|any|this)\s+(?:have|support|do|provide)?\s*(?:power\s+delivery|pd|usb-?pd|charging?|charge)\b',
                r'\bpower\s+delivery\b',
                r'\b(?:laptop\s+)?charging?\b',
                r'\bcharge\s+(?:my\s+)?(?:laptop|device|macbook)\b',
            ],
            'field': 'features',
            'value': 'Power Delivery',
        },
        # 8K support
        '8k': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|this|these|those|any)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*8k\b',
                r'\b8k\s+(?:support|compatible|capable)\b',
            ],
            'field': 'features',
            'value': '8K',
        },
        # 1080p / Full HD support
        '1080p': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|this|these|those|any)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*(?:1080p?|full\s*hd)\b',
                r'\b(?:1080p?|full\s*hd)\s+(?:support|compatible|capable)\b',
            ],
            'field': 'features',
            'value': '1080p',
        },
        # 1440p / 2K / QHD support
        '1440p': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|this|these|those|any)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*(?:1440p?|2k|qhd)\b',
                r'\b(?:1440p?|2k|qhd)\s+(?:support|compatible|capable)\b',
            ],
            'field': 'features',
            'value': '1440p',
        },
        # HDR support
        'hdr': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|this|these|those|any)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*hdr\b',
                r'\bhdr\s+(?:support|compatible|capable|10)\b',
            ],
            'field': 'features',
            'value': 'HDR',
        },
        # USB 3.0 speed
        'usb_3_0': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*usb\s*3\.?0\b',
                r'\busb\s*3\.?0\s+(?:support|compatible|capable|speed)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+usb\s*3\.?0\b',
            ],
            'field': 'features',
            'value': 'USB 3.0',
        },
        # USB 3.1 speed
        'usb_3_1': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*usb\s*3\.?1\b',
                r'\busb\s*3\.?1\s+(?:support|compatible|capable|speed)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+usb\s*3\.?1\b',
            ],
            'field': 'features',
            'value': 'USB 3.1',
        },
        # USB 3.2 speed
        'usb_3_2': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*usb\s*3\.?2\b',
                r'\busb\s*3\.?2\s+(?:support|compatible|capable|speed)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+usb\s*3\.?2\b',
            ],
            'field': 'features',
            'value': 'USB 3.2',
        },
        # USB 2.0 speed
        'usb_2_0': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*usb\s*2\.?0\b',
                r'\busb\s*2\.?0\s+(?:support|compatible|capable|speed)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+usb\s*2\.?0\b',
            ],
            'field': 'features',
            'value': 'USB 2.0',
        },
        # HDCP (copy protection)
        'hdcp': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*hdcp\b',
                r'\bhdcp\s+(?:support|compatible|capable|compliant|2\.2|2\.3)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+hdcp\b',
            ],
            'field': 'features',
            'value': 'HDCP',
        },
        # eARC (Enhanced Audio Return Channel) - MUST come before ARC (earc contains arc)
        'earc': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*(?:enhanced\s+audio\s+return\s+channel|earc|e-?arc)\b',
                r'\b(?:enhanced\s+audio\s+return\s+channel|earc|e-?arc)\s+(?:support|compatible|capable)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+(?:earc|e-?arc)\b',
            ],
            'field': 'features',
            'value': 'eARC',
        },
        # ARC (Audio Return Channel)
        'arc': {
            'patterns': [
                r'\b(?:do|does|are|is)\s+(?:they|it|these|those|any|this)\s+(?:all\s+)?(?:support|have|work\s+with)?\s*(?:audio\s+return\s+channel|arc)\b',
                r'\b(?:audio\s+return\s+channel|arc)\s+(?:support|compatible|capable)\b',
                r'\b(?:is|are)\s+(?:this|these|it|they)\s+arc\b',
            ],
            'field': 'features',
            'value': 'ARC',
        },
    }

    def _handle_yes_no_question(self, query: str, products: List[Product]) -> Optional[str]:
        """
        Handle yes/no questions about product attributes.

        Answers questions like:
        - "Are they both red?"
        - "Do any of these support 4K?"
        - "Are these cables black?"

        Returns the answer with a direct yes/no first, then details.
        """
        # Check for color questions
        color_response = self._handle_color_yes_no(query, products)
        if color_response:
            return color_response

        # Check for feature questions (4K, Thunderbolt, etc.)
        feature_response = self._handle_feature_yes_no(query, products)
        if feature_response:
            return feature_response

        return None

    def _handle_color_yes_no(self, query: str, products: List[Product]) -> Optional[str]:
        """Handle yes/no questions specifically about color."""
        config = self.YES_NO_ATTRIBUTE_PATTERNS['color']

        # Check if this is a color question
        asked_color = None
        for pattern in config['patterns']:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_color = match.group(1).lower()
                if potential_color in config['color_words']:
                    asked_color = potential_color
                    break

        if not asked_color:
            return None

        # Get colors for all products
        product_colors = []
        for i, prod in enumerate(products, 1):
            color = prod.metadata.get('color', '')
            product_colors.append({
                'index': i,
                'sku': prod.product_number,
                'color': color,
                'color_lower': color.lower() if color else '',
            })

        # Determine the answer
        matches = [p for p in product_colors if asked_color in p['color_lower']]
        all_match = len(matches) == len(products)
        none_match = len(matches) == 0
        some_match = 0 < len(matches) < len(products)

        lines = []

        if all_match:
            # All products match the color
            if len(products) == 1:
                lines.append(f"**Yes**, the **{product_colors[0]['sku']}** is {asked_color.title()}.")
            else:
                lines.append(f"**Yes**, all {len(products)} products are {asked_color.title()}.")
        elif none_match:
            # No products match
            if len(products) == 1:
                actual_color = product_colors[0]['color'] or 'unknown color'
                lines.append(f"**No**, the **{product_colors[0]['sku']}** is {actual_color}, not {asked_color}.")
            else:
                lines.append(f"**No**, none of these are {asked_color}. Here are the actual colors:")
                lines.append("")
                for p in product_colors:
                    color_display = p['color'] if p['color'] else 'Color not specified'
                    lines.append(f"- **{p['sku']}**: {color_display}")
        else:
            # Some match, some don't
            lines.append(f"**Some are, some aren't:**")
            lines.append("")
            for p in product_colors:
                is_match = asked_color in p['color_lower']
                color_display = p['color'] if p['color'] else 'Color not specified'
                status = "✓" if is_match else "✗"
                lines.append(f"- {status} **{p['sku']}**: {color_display}")

        # Add helpful follow-up
        if none_match:
            lines.append("")
            lines.append(f"Would you like me to search specifically for {asked_color} products?")

        return "\n".join(lines)

    def _handle_feature_yes_no(self, query: str, products: List[Product]) -> Optional[str]:
        """Handle yes/no questions about features (4K, Thunderbolt, etc.)."""

        # Special handling for "USB X or USB Y" questions - report what it IS
        usb_or_pattern = r'\busb\s*(\d\.?\d?)\s+or\s+usb\s*(\d\.?\d?)\b'
        usb_or_match = re.search(usb_or_pattern, query, re.IGNORECASE)
        if usb_or_match:
            return self._answer_usb_version_question(products)

        # Special handling for "what USB version" questions
        usb_what_pattern = r'\b(?:what|which)\s+usb\s+(?:version|type|speed)\b'
        if re.search(usb_what_pattern, query, re.IGNORECASE):
            return self._answer_usb_version_question(products)

        for attr_name, config in self.YES_NO_ATTRIBUTE_PATTERNS.items():
            if attr_name == 'color':
                continue  # Handled separately

            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    return self._answer_feature_question(products, config['field'], config['value'], attr_name)

        return None

    def _answer_usb_version_question(self, products: List[Product]) -> str:
        """Answer questions about what USB version a product is."""
        results = []
        for i, prod in enumerate(products, 1):
            usb_version = self._get_actual_usb_version(prod)
            results.append({
                'index': i,
                'sku': prod.product_number,
                'usb_version': usb_version,
            })

        # Check if all products have the same USB version
        versions = [r['usb_version'] for r in results if r['usb_version']]

        if not versions:
            return "USB version information is not available for these products."

        unique_versions = set(versions)

        if len(unique_versions) == 1:
            version = list(unique_versions)[0]
            if len(products) == 1:
                return f"The **{results[0]['sku']}** is **{version}**."
            else:
                return f"All {len(products)} products are **{version}**."
        else:
            lines = ["These products have different USB versions:"]
            lines.append("")
            for r in results:
                version = r['usb_version'] or "Unknown"
                lines.append(f"- **{r['sku']}**: {version}")
            return "\n".join(lines)

    def _get_inherent_capability(self, product: Product, feature: str) -> Optional[str]:
        """
        Check if a cable type inherently supports a feature based on connector type.

        Returns:
            - None if no inherent capability known
            - A string explanation if the cable inherently supports the feature
        """
        feature_lower = feature.lower()
        content = (product.content or '').lower()
        name = product.metadata.get('name', '').lower()
        connectors = product.metadata.get('connectors', [])
        connector_str = ' '.join(str(c).lower() for c in connectors)
        sub_category = product.metadata.get('sub_category', '').lower()
        category = product.metadata.get('category', '').lower()

        # DisplayPort cables inherently support high resolutions
        if 'displayport' in connector_str or 'dp cable' in sub_category:
            if feature_lower in ('4k', '1440p', '1080p', '2k', 'qhd', '8k'):
                return f"DisplayPort cables support {feature} and higher resolutions by design"

        # HDMI High Speed with Ethernet supports ARC
        if ('hdmi' in connector_str or 'hdmi' in sub_category) and 'ethernet' in content:
            if feature_lower in ('arc', 'audio return channel'):
                return "High Speed HDMI with Ethernet cables support ARC by design"

        # HDMI cables generally support common resolutions
        if 'hdmi' in connector_str or 'hdmi' in sub_category:
            if feature_lower in ('1080p', '1440p', '4k'):
                return f"HDMI cables support {feature} resolution"
            if feature_lower == 'hdcp':
                return "HDMI cables support HDCP copy protection"

        # Thunderbolt cables support high resolutions and data speeds
        if 'thunderbolt' in connector_str or 'thunderbolt' in sub_category:
            if feature_lower in ('4k', '1440p', '1080p', '8k'):
                return f"Thunderbolt cables support {feature} and higher resolutions"

        return None

    def _get_actual_usb_version(self, product: Product) -> Optional[str]:
        """Get the actual USB version from product data."""
        usb_type = product.metadata.get('usb_type', '') or product.metadata.get('USBTYPE', '')
        content = (product.content or '').lower()

        if usb_type:
            return str(usb_type)

        # Try to infer from content
        if 'usb 3.2' in content or 'usb3.2' in content:
            return 'USB 3.2'
        if 'usb 3.1' in content or 'usb3.1' in content:
            return 'USB 3.1'
        if 'usb 3.0' in content or 'usb3.0' in content or '5gbps' in content:
            return 'USB 3.0'
        if 'usb 2.0' in content or 'usb2.0' in content or '480' in content:
            return 'USB 2.0'

        return None

    def _answer_feature_question(
        self,
        products: List[Product],
        field: str,
        value: str,
        display_name: str
    ) -> Optional[str]:
        """Generate answer for a feature yes/no question."""
        # Check which products have this feature
        results = []
        for i, prod in enumerate(products, 1):
            has_feature = False
            inherent_reason = None

            # Special handling for Power Delivery - check multiple metadata fields
            if display_name == 'power_delivery' or value.lower() == 'power delivery':
                # Power Delivery can be in features list, power_delivery, or hub_power_delivery fields
                pd_value = prod.metadata.get('power_delivery') or prod.metadata.get('hub_power_delivery')
                if pd_value and str(pd_value).strip() and str(pd_value).lower() not in ('no', 'nan', ''):
                    has_feature = True
                # Also check features list and content for "Power Delivery" or "charging"
                if not has_feature:
                    features = prod.metadata.get('features', [])
                    has_feature = any('power delivery' in f.lower() or 'charging' in f.lower() for f in features)
                # Check product content for charging mentions
                if not has_feature:
                    content = (prod.content or '').lower()
                    has_feature = 'power delivery' in content or ('charging' in content and 'w' in content)
            elif field == 'features':
                features = prod.metadata.get('features', [])
                # Case-insensitive feature matching
                has_feature = any(value.lower() in f.lower() for f in features)

                # If not found explicitly, check inherent capabilities
                if not has_feature:
                    inherent_reason = self._get_inherent_capability(prod, value)
                    if inherent_reason:
                        has_feature = True
            else:
                field_value = prod.metadata.get(field, '')
                has_feature = value.lower() in str(field_value).lower()

            results.append({
                'index': i,
                'sku': prod.product_number,
                'has_feature': has_feature,
                'inherent_reason': inherent_reason,
            })

        matches = [r for r in results if r['has_feature']]
        all_match = len(matches) == len(products)
        none_match = len(matches) == 0

        # Check if any results have inherent capability (not explicit data)
        has_inherent = any(r.get('inherent_reason') for r in results if r['has_feature'])

        lines = []

        if all_match:
            if len(products) == 1:
                lines.append(f"**Yes**, the **{results[0]['sku']}** supports {value}.")
            else:
                lines.append(f"**Yes**, all {len(products)} products support {value}.")

            # Add inherent capability explanation if applicable
            if has_inherent:
                inherent_reasons = set(r['inherent_reason'] for r in results if r.get('inherent_reason'))
                if inherent_reasons:
                    lines.append("")
                    lines.append(f"*{list(inherent_reasons)[0]}.*")
        elif none_match:
            if len(products) == 1:
                lines.append(f"**No**, the **{results[0]['sku']}** does not explicitly list {value} support.")
            else:
                lines.append(f"**No**, none of these products explicitly list {value} support in their specs.")
        else:
            lines.append(f"**Some do, some don't:**")
            lines.append("")
            for r in results:
                status = "✓ Yes" if r['has_feature'] else "✗ No"
                reason = f" ({r['inherent_reason']})" if r.get('inherent_reason') else ""
                lines.append(f"- **{r['sku']}**: {status}{reason}")

        return "\n".join(lines)


# Singleton instance
_followup_handler = FollowupHandler()


def get_followup_handler() -> FollowupHandler:
    """Get the follow-up handler instance."""
    return _followup_handler
