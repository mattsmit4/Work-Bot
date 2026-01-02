"""
Follow-up intent handlers.

Handles follow-up questions about products in context:
- Multi-followup: Questions about multiple products
- Single-followup: Questions about a single product
- Refinement: "I need 3 foot cables instead"
"""

import re
from handlers.base import BaseHandler, HandlerContext, HandlerResult
from core.product_validator import get_best_cable
from ui.responses import format_dock_specs
from llm.followup_handler import get_followup_handler
from llm.technical_question_handler import TechnicalQuestionHandler


class FollowupHandler(BaseHandler):
    """Handle follow-up questions about products in context."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        # Check for refinement (e.g., "I need 3 foot cables")
        if ctx.intent.meta_info and ctx.intent.meta_info.get('refinement'):
            return self._handle_refinement(ctx)

        # Check if we have products in context
        if not ctx.context.current_products:
            return HandlerResult(
                response="I can help you find StarTech.com products. What are you looking for?"
            )

        # Try followup handler (comparisons, specific product questions)
        followup_handler = get_followup_handler()
        answer = followup_handler.handle_followup(
            query=ctx.query,
            products=ctx.context.current_products,
            intent=ctx.intent,
            context=ctx.context
        )

        if answer:
            return HandlerResult(response=answer)

        # Try technical question handler
        tech_handler = TechnicalQuestionHandler()
        tech_answer = tech_handler.answer_technical_question(
            query=ctx.query,
            products=ctx.context.current_products
        )

        if tech_answer:
            return HandlerResult(response=tech_answer)

        # Fallback: show product specs
        return self._show_product_specs(ctx)

    def _handle_refinement(self, ctx: HandlerContext) -> HandlerResult:
        """Handle refinement requests (e.g., different length)."""
        if not ctx.context.current_products:
            return HandlerResult(
                response="I don't have any products to refine. What are you looking for?"
            )

        query_lower = ctx.query.lower()

        # Check for relative length refinement (shorter/longer)
        if re.search(r'\bshorter\b', query_lower):
            return self._refine_by_relative_length(ctx, 'shorter')
        if re.search(r'\blonger\b', query_lower):
            return self._refine_by_relative_length(ctx, 'longer')

        # Extract new filters for absolute length
        new_filters = ctx.filter_extractor.extract(ctx.query)
        requested_length = new_filters.length
        length_unit = new_filters.length_unit or 'ft'

        if requested_length:
            return self._refine_by_length(ctx, requested_length, length_unit)

        # Check for feature-based refinement
        keywords = self._extract_requirement_keywords(ctx.query)
        if keywords:
            return self._refine_by_requirements(ctx, keywords)

        return HandlerResult(
            response="I couldn't determine what you're looking for. "
                     "Could you specify what features or length you need?"
        )

    def _refine_by_length(self, ctx: HandlerContext, length: float, unit: str) -> HandlerResult:
        """Re-search for products with new length."""
        ctx.add_debug(f"📏 REFINEMENT: length={length}{unit}")

        # Collect unique connector pairs from context
        unique_pairs = {}
        for prod in ctx.context.current_products:
            connectors = prod.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                source = self._normalize_connector(connectors[0])
                target = self._normalize_connector(connectors[1])
                if source:
                    key = (source, target or source)
                    if key not in unique_pairs:
                        unique_pairs[key] = prod

        parts = [f"Here are {int(length)}{unit} options for your setup:", ""]
        all_products = []

        for (source, target), _ in unique_pairs.items():
            filters = ctx.filter_extractor.extract("")
            filters.connector_from = source
            filters.connector_to = target
            filters.length = length
            filters.length_unit = unit

            results = ctx.search_engine.search(filters)

            cable_desc = f"{source} to {target}" if source != target else f"{source} cable"
            parts.append(f"**{cable_desc}:**")

            if results.products:
                length_ft = length if unit == 'ft' else length * 3.28
                best = get_best_cable(results.products, source, target, preferred_length_ft=length_ft)

                if best:
                    all_products.append(best)
                    name = best.metadata.get('name', best.product_number)
                    length_display = best.metadata.get('length_display', '')
                    parts.append(f"Recommended: **{name}** ({best.product_number})")
                    if length_display:
                        parts.append(f"Length: {length_display}")
                else:
                    parts.append(f"_No {int(length)}{unit} cable found_")
            else:
                parts.append(f"_No {int(length)}{unit} version available_")
            parts.append("")

        parts.append("Would you like more details on any of these?")

        return HandlerResult(
            response="\n".join(parts),
            products_to_set=all_products if all_products else None
        )

    def _refine_by_relative_length(self, ctx: HandlerContext, direction: str) -> HandlerResult:
        """
        Handle relative length refinement (shorter/longer).

        Args:
            ctx: Handler context
            direction: 'shorter' or 'longer'
        """
        ctx.add_debug(f"📏 RELATIVE REFINEMENT: {direction}")

        # Get current product lengths
        current_lengths = []
        for prod in ctx.context.current_products:
            length_ft = prod.metadata.get('length_ft')
            if length_ft:
                current_lengths.append((prod, length_ft))

        if not current_lengths:
            return HandlerResult(
                response="I don't have length information for these products. "
                         "Could you specify what length you need?"
            )

        # Find the reference length (shortest for "shorter", longest for "longer")
        if direction == 'shorter':
            reference_length = min(l for _, l in current_lengths)
        else:
            reference_length = max(l for _, l in current_lengths)

        ctx.add_debug(f"📏 Reference length: {reference_length}ft, looking for {direction}")

        # Get connector info from current products
        unique_pairs = {}
        for prod in ctx.context.current_products:
            connectors = prod.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                source = self._normalize_connector(connectors[0])
                target = self._normalize_connector(connectors[1])
                if source:
                    key = (source, target or source)
                    if key not in unique_pairs:
                        unique_pairs[key] = prod

        if not unique_pairs:
            return HandlerResult(
                response="I couldn't determine the connector types. "
                         "Could you tell me what type of cable you need?"
            )

        # Search for products with the desired length
        all_matches = []
        for (source, target), _ in unique_pairs.items():
            filters = ctx.filter_extractor.extract("")
            filters.connector_from = source
            filters.connector_to = target

            results = ctx.search_engine.search(filters)

            for prod in results.products:
                length_ft = prod.metadata.get('length_ft')
                if length_ft:
                    if direction == 'shorter' and length_ft < reference_length:
                        all_matches.append((prod, length_ft))
                    elif direction == 'longer' and length_ft > reference_length:
                        all_matches.append((prod, length_ft))

        if not all_matches:
            if direction == 'shorter':
                return HandlerResult(
                    response=f"The current products are already the shortest available. "
                             f"The shortest is {reference_length}ft."
                )
            else:
                return HandlerResult(
                    response=f"The current products are already the longest available. "
                             f"The longest is {reference_length}ft."
                )

        # Sort by length
        if direction == 'shorter':
            all_matches.sort(key=lambda x: -x[1])  # Longest first (closest to current)
        else:
            all_matches.sort(key=lambda x: x[1])   # Shortest first (closest to current)

        # Get top 3 unique products
        seen_skus = set()
        top_products = []
        for prod, length in all_matches:
            if prod.product_number not in seen_skus:
                seen_skus.add(prod.product_number)
                top_products.append(prod)
                if len(top_products) >= 3:
                    break

        # Build response
        direction_word = "shorter" if direction == 'shorter' else "longer"
        parts = [f"Here are {direction_word} options:", ""]

        for i, prod in enumerate(top_products, 1):
            name = prod.metadata.get('name', prod.product_number)
            length_display = prod.metadata.get('length_display', '')
            parts.append(f"{i}. **{prod.product_number}** - {name}")
            if length_display:
                parts.append(f"   Length: {length_display}")
            parts.append("")

        parts.append("Would you like more details on any of these?")

        return HandlerResult(
            response="\n".join(parts),
            products_to_set=top_products
        )

    def _refine_by_requirements(self, ctx: HandlerContext, keywords: list) -> HandlerResult:
        """Filter products by requirement keywords."""
        ctx.add_debug(f"🎯 REFINEMENT: keywords={keywords}")

        # Score products
        scored = []
        for prod in ctx.context.current_products:
            score = self._score_by_requirements(prod, keywords)
            scored.append((prod, score))

        scored.sort(key=lambda x: (x[1], x[0].score), reverse=True)
        matches = [(p, s) for p, s in scored if s > 0]

        if not matches:
            return HandlerResult(
                response=f"None of the current products match your requirements. "
                         f"Would you like me to search for products with {', '.join(keywords[:3])}?"
            )

        parts = [f"Based on your requirements ({', '.join(keywords[:4])}), here are the best options:", ""]

        shown = 0
        for prod, score in matches[:5]:
            if shown >= 3 and score < matches[0][1]:
                break

            name = prod.metadata.get('name', prod.product_number)
            category = prod.metadata.get('category', '')

            parts.append(f"**{name}** ({prod.product_number})")

            if category in ('dock', 'hub'):
                specs = format_dock_specs(prod)
                for spec in specs:
                    parts.append(f"   - {spec}")
            else:
                features = prod.metadata.get('features', [])
                if features:
                    parts.append(f"   Features: {', '.join(features[:5])}")

            parts.append("")
            shown += 1

        parts.append("Would you like more details on any of these?")

        return HandlerResult(
            response="\n".join(parts),
            products_to_set=[p for p, _ in matches[:5]]
        )

    def _show_product_specs(self, ctx: HandlerContext) -> HandlerResult:
        """Show full specs for products in context."""
        response = "Here are the full specs:\n\n"

        for i, prod in enumerate(ctx.context.current_products, 1):
            category = prod.metadata.get('category', '').lower()
            name = prod.metadata.get('name', prod.product_number)

            response += f"**{i}. {name}** ({prod.product_number})\n\n"

            if category in ('dock', 'hub', 'docking_station'):
                specs = format_dock_specs(prod)
                for spec in specs:
                    response += f"- {spec}\n"
            else:
                length = prod.metadata.get('length_display')
                if length:
                    response += f"- Length: {length}\n"

                connectors = prod.metadata.get('connectors')
                if connectors and len(connectors) >= 2:
                    response += f"- Connectors: {connectors[0]} → {connectors[1]}\n"

                features = prod.metadata.get('features', [])
                response += f"- Features: {', '.join(features) if features else 'Standard features'}\n"

            response += "\n"

        response += "Would you like me to compare any of these, or do you have specific questions?"
        return HandlerResult(response=response)

    def _normalize_connector(self, conn: str) -> str | None:
        """Normalize connector name to standard form."""
        conn_lower = conn.lower()
        mapping = {
            'hdmi': 'HDMI', 'usb-c': 'USB-C', 'usb c': 'USB-C',
            'type-c': 'USB-C', 'displayport': 'DisplayPort', 'dp': 'DisplayPort',
            'vga': 'VGA', 'dvi': 'DVI'
        }
        for key, value in mapping.items():
            if key in conn_lower:
                return value
        return None

    def _extract_requirement_keywords(self, query: str) -> list:
        """Extract requirement keywords from query."""
        query_lower = query.lower()
        keywords = []

        # Monitor patterns
        if re.search(r'\b(?:dual|2|two)\s*monitors?\b', query_lower):
            keywords.append('dual monitor')
        if re.search(r'\b(?:triple|3|three)\s*monitors?\b', query_lower):
            keywords.append('triple monitor')

        # Video features
        if re.search(r'\b4k\b', query_lower):
            keywords.append('4K')
        if re.search(r'\b60\s*hz\b', query_lower):
            keywords.append('60Hz')

        # Power/charging
        if re.search(r'\bcharg(?:e|ing)\b|\bpower\s*delivery\b|\bpd\b', query_lower):
            keywords.append('power delivery')

        # Connectivity
        if 'ethernet' in query_lower:
            keywords.append('ethernet')
        if re.search(r'\busb[\s-]?a\b', query_lower):
            keywords.append('USB-A')
        if 'sd card' in query_lower:
            keywords.append('SD card')

        return list(set(keywords))

    def _score_by_requirements(self, product, requirements: list) -> int:
        """Score product by how many requirements it matches."""
        score = 0
        content_lower = product.content.lower()
        features_lower = [f.lower() for f in product.metadata.get('features', [])]
        meta = product.metadata

        for req in requirements:
            req_lower = req.lower()

            # 4K check - works for all product types (cables, docks, adapters)
            # ONLY use supports_4k(), don't fall through to text matching
            if req_lower == '4k':
                if product.supports_4k():
                    score += 1
                continue  # Skip text matching for 4K - it's a specific technical check

            # Dock-specific checks
            if meta.get('category') in ('dock', 'hub'):
                if req_lower == 'power delivery':
                    if meta.get('power_delivery') or meta.get('hub_power_delivery'):
                        score += 1
                        continue
                if req_lower == 'ethernet':
                    if meta.get('network_speed') or 'RJ-45' in meta.get('CONNTYPE', ''):
                        score += 1
                        continue

            # Text matching
            if req_lower in content_lower:
                score += 1
            elif any(req_lower in f for f in features_lower):
                score += 1

        return score
