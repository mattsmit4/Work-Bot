"""
Search intent handlers - Simplified MVP.

Handles new product searches with simple, reliable logic.
"""

import re
from handlers.base import BaseHandler, HandlerContext, HandlerResult
from core.context import SearchFilters, Product


class NewSearchHandler(BaseHandler):
    """Handle new product search queries."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        # Clear stale context
        self._clear_stale_context(ctx)

        # Check for direct SKU lookup
        if ctx.intent.sku:
            return self._handle_sku_lookup(ctx, ctx.intent.sku)

        # Extract filters from query
        filters = ctx.filter_extractor.extract(ctx.query)
        ctx.add_debug(f"🔍 FILTERS: {filters}")

        # Perform search
        results = ctx.search_engine.search(filters)
        ctx.add_debug(f"🔍 SEARCH: Found {len(results.products)} products")

        # Apply port type filtering for docks/hubs
        if filters.required_port_types:
            original_count = len(results.products)
            results.products = self._filter_by_port_types(
                results.products, filters.required_port_types
            )
            ctx.add_debug(
                f"🔍 PORT FILTER: {filters.required_port_types} → "
                f"{original_count} → {len(results.products)} products"
            )

        if not results.products:
            # Try fallback searches
            return self._handle_no_results(ctx, filters)

        # Rank products
        filters_dict = {
            'length': filters.length,
            'features': filters.features,
            'length_preference': getattr(filters, 'length_preference', None),
        }

        ranked_products = ctx.product_ranker.get_top_n(
            products=results.products,
            query=ctx.query,
            n=3,
            extracted_filters=filters_dict
        )

        # Build response
        original_filters_dict = {
            'color': filters.color,
            'length': filters.length,
            'connector_from': filters.connector_from,
            'connector_to': filters.connector_to,
        }
        response = ctx.response_builder.build_response(
            ranked_products=ranked_products,
            query=ctx.query,
            dropped_filters=results.dropped_filters,
            original_filters=original_filters_dict,
        )

        top_products = [rp.product for rp in ranked_products]
        return HandlerResult(
            response=response,
            products_to_set=top_products
        )

    def _handle_no_results(self, ctx: HandlerContext, filters: SearchFilters) -> HandlerResult:
        """Handle case when initial search returns no results."""
        # Try relaxed search - drop length and features
        from core.context import SearchFilters as FallbackFilters

        fallback_note = None
        fallback_results = None

        # Fallback 1: Drop length and features, keep connectors
        if filters.connector_from or filters.connector_to:
            relaxed = FallbackFilters()
            relaxed.connector_from = filters.connector_from
            relaxed.connector_to = filters.connector_to
            relaxed.product_category = filters.product_category

            ctx.add_debug(f"🔄 FALLBACK 1: Relaxed search without length/features")
            fallback_results = ctx.search_engine.search(relaxed)

            if fallback_results.products:
                conn_from = filters.connector_from or "?"
                conn_to = filters.connector_to or "?"
                if filters.length:
                    fallback_note = f"**Note:** No {int(filters.length)}ft {conn_from} to {conn_to} cables found. Here are available options:"
                elif filters.features:
                    feat_str = ', '.join(filters.features)
                    fallback_note = f"**Note:** No {conn_from} to {conn_to} cables with {feat_str} found. Here are available options:"

        # Fallback 2: Try just one connector
        if not fallback_results or not fallback_results.products:
            for connector in [filters.connector_from, filters.connector_to]:
                if connector:
                    simple = FallbackFilters()
                    simple.connector_from = connector
                    simple.product_category = filters.product_category or 'Cables'

                    ctx.add_debug(f"🔄 FALLBACK 2: Single connector search: {connector}")
                    fallback_results = ctx.search_engine.search(simple)

                    if fallback_results.products:
                        conn_from = filters.connector_from or "?"
                        conn_to = filters.connector_to or "?"
                        fallback_note = f"**Note:** No {conn_from} to {conn_to} products found. Here are {connector} products:"
                        break

        if fallback_results and fallback_results.products:
            # Use fallback results
            ranked = ctx.product_ranker.get_top_n(
                products=fallback_results.products,
                query=ctx.query,
                n=3
            )

            response = ctx.response_builder.build_response(
                ranked_products=ranked,
                query=ctx.query,
                intro_text=fallback_note + "\n" if fallback_note else None,
            )

            top_products = [rp.product for rp in ranked]
            return HandlerResult(
                response=response,
                products_to_set=top_products
            )

        # All fallbacks failed
        suggestions = [
            "Try searching by connector type (USB-C, HDMI, DisplayPort)",
            "Try a different cable length",
            "Use simpler terms like 'USB-C cable' or 'HDMI adapter'"
        ]
        response = ctx.formatter.format_no_results(ctx.query, suggestions)
        return HandlerResult(response=response)

    def _filter_by_port_types(
        self,
        products: list[Product],
        required_port_types: list[str]
    ) -> list[Product]:
        """
        Filter products to only those that have all required port types.

        Args:
            products: List of products to filter
            required_port_types: Port types the product must have (e.g., ["USB-C"])

        Returns:
            Filtered list of products that have all required port types
        """
        if not required_port_types:
            return products

        return [p for p in products if self._product_has_port_types(p, required_port_types)]

    def _product_has_port_types(
        self,
        product: Product,
        required_port_types: list[str]
    ) -> bool:
        """
        Check if a product has all the required port types.

        Checks the CONNTYPE field in product metadata for port type mentions.
        For docks, CONNTYPE contains entries like "1 x USB 3.0 Type-C",
        "2 x USB Type-A", etc.

        Args:
            product: Product to check
            required_port_types: Port types to look for (e.g., ["USB-C", "USB-A"])

        Returns:
            True if product has all required port types
        """
        # Get CONNTYPE field (contains port information)
        conntype = product.metadata.get('CONNTYPE', '')
        if not conntype:
            # Also check 'conntype' (lowercase) as fallback
            conntype = product.metadata.get('conntype', '')

        if not conntype:
            # No port info - can't verify, exclude from filtered results
            return False

        conntype_lower = str(conntype).lower()

        # Port type patterns to match in CONNTYPE
        port_patterns = {
            'USB-C': [r'type[\s\-]?c', r'usb[\s\-]?c', r'usb\s+3\.\d+\s+type-c'],
            'USB-A': [r'type[\s\-]?a', r'usb[\s\-]?a', r'usb\s+\d+\.\d+\s+type-a'],
            'USB': [r'\busb\b'],  # Generic USB
            'HDMI': [r'\bhdmi\b'],
            'DisplayPort': [r'\bdisplayport\b', r'\bdp\b'],
            'Thunderbolt': [r'\bthunderbolt\b'],
            'Ethernet': [r'\bethernet\b', r'\brj[\s\-]?45\b', r'\bgigabit\b'],
        }

        # Check if all required port types are present
        for port_type in required_port_types:
            patterns = port_patterns.get(port_type, [port_type.lower()])

            # Check if any pattern matches
            found = False
            for pattern in patterns:
                if re.search(pattern, conntype_lower):
                    found = True
                    break

            if not found:
                return False

        return True

    def _handle_sku_lookup(self, ctx: HandlerContext, sku: str) -> HandlerResult:
        """
        Handle direct SKU lookup.

        Args:
            ctx: Handler context
            sku: Product SKU to look up

        Returns:
            HandlerResult with product info or not found message
        """
        ctx.add_debug(f"🔍 SKU LOOKUP: {sku}")

        # Search for the exact SKU in all products
        matching_products = []
        sku_upper = sku.upper()

        for product in ctx.all_products:
            product_sku = product.product_number.upper()
            # Exact match or starts with (for partial SKUs)
            if product_sku == sku_upper or product_sku.startswith(sku_upper):
                matching_products.append(product)

        if not matching_products:
            ctx.add_debug(f"🔍 SKU LOOKUP: No products found for {sku}")
            return HandlerResult(
                response=f"I couldn't find a product with SKU **{sku}**. "
                         f"Please check the SKU and try again, or describe what you're looking for."
            )

        ctx.add_debug(f"🔍 SKU LOOKUP: Found {len(matching_products)} products")

        # If exact match, show that product
        exact_match = next((p for p in matching_products if p.product_number.upper() == sku_upper), None)

        if exact_match:
            # Single product found - show detailed info
            product = exact_match
            meta = product.metadata
            name = meta.get('name', product.product_number)
            category = meta.get('category', '')
            length = meta.get('length_display', '')
            connectors = meta.get('connectors', [])
            features = meta.get('features', [])
            warranty = meta.get('warranty', '')

            parts = [f"Here's **{product.product_number}** - {name}:", ""]

            if category:
                parts.append(f"**Category:** {category}")
            if length:
                parts.append(f"**Length:** {length}")
            if connectors and len(connectors) >= 2:
                parts.append(f"**Connectors:** {connectors[0]} to {connectors[1]}")
            if features:
                parts.append(f"**Features:** {', '.join(features[:5])}")
            if warranty:
                parts.append(f"**Warranty:** {warranty}")

            parts.append("")
            parts.append("Would you like to know more about this product, or compare it with similar options?")

            return HandlerResult(
                response="\n".join(parts),
                products_to_set=[product]
            )

        # Multiple partial matches - show list
        response_parts = [f"I found {len(matching_products)} products matching **{sku}**:", ""]
        for i, product in enumerate(matching_products[:5], 1):
            name = product.metadata.get('name', product.product_number)
            response_parts.append(f"{i}. **{product.product_number}** - {name}")

        response_parts.append("")
        response_parts.append("Which one would you like to know more about?")

        return HandlerResult(
            response="\n".join(response_parts),
            products_to_set=matching_products[:5]
        )
