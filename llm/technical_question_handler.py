"""
Technical Question Handler - Answer spec questions about products in context

Detects when users ask technical questions about products and provides
factual answers based on actual product data from Excel.

Examples:
- "What's the max data rate transfer?" → "10.2 Gbps for High-Speed HDMI"
- "Do they support 4K?" → "Yes, all three support 4K at 30Hz"
- "How long are they?" → "All three are 15 feet"
"""

import re
from typing import List, Optional, Tuple
from core.context import Product


class TechnicalQuestionHandler:
    """
    Handles technical specification questions about products in context.
    """
    
    def __init__(self):
        """Initialize with question patterns."""
        
        # Question patterns and their corresponding handlers
        # Order matters: more specific patterns should come BEFORE generic ones
        self.question_patterns = [
            # === SPECIFIC PATTERNS FIRST (before generic "do...have" patterns) ===

            # Gold plating (extended specs) - check BEFORE generic features
            (r'gold[\s\-]?plat', 'plating'),
            (r'plat(?:ed|ing)', 'plating'),
            (r'connector\s*(?:material|quality)', 'plating'),

            # Detailed resolution (60Hz vs 30Hz) - check BEFORE generic resolution
            (r'(?:60|30|120)\s*hz', 'resolution_detail'),
            (r'refresh\s*rate', 'resolution_detail'),

            # Wire gauge / thickness (extended specs)
            (r'(?:wire\s*)?gauge', 'wire_gauge'),
            (r'awg', 'wire_gauge'),
            (r'thick(?:ness|er|est)?', 'wire_gauge'),
            (r'(?:for|over)\s*long\s*(?:distance|run)', 'wire_gauge'),
            (r'long\s*(?:cable\s*)?run', 'wire_gauge'),
            (r'best\s*(?:for|over)\s*(?:long|distance)', 'wire_gauge'),

            # Audio/headphone questions
            (r'audio|headphone|head\s*phone|3\.5\s*mm|jack|sound', 'features'),

            # === GENERAL PATTERNS ===

            # Data rate / bandwidth / speed questions
            (r'(?:max|maximum)?\s*(?:data\s*rate|transfer\s*rate|bandwidth|speed)', 'data_rate'),
            (r'(?:how\s*)?fast', 'data_rate'),
            (r'gbps|mbps', 'data_rate'),

            # Resolution questions (general)
            (r'(?:support|handle|do).*?(?:4k|8k|1080p|resolution)', 'resolution'),
            (r'what\s*resolution', 'resolution'),

            # Length questions
            (r'(?:how\s*)?long', 'length'),
            (r'length', 'length'),

            # Connector questions
            (r'(?:what|which)\s*connectors?', 'connectors'),
            (r'connector\s*type', 'connectors'),

            # Active vs passive
            (r'active\s*(?:or\s*passive)?', 'active_passive'),
            (r'(?:is|are)\s*(?:it|they|these)\s*active', 'active_passive'),

            # Shielded
            (r'shielded', 'shielded'),

            # Color
            (r'(?:what|which)\s*color', 'color'),
            (r'\bcolor\b', 'color'),

            # Warranty
            (r'warranty', 'warranty'),
            (r'(?:how\s*long|what)\s*(?:is|does).*(?:cover|guarantee)', 'warranty'),

            # Feature questions (generic - LAST so specific patterns match first)
            (r'(?:do|does|can).*?(?:support|have)', 'features'),
            (r'(?:is|are)\s*(?:it|they|these)\s*(?:compatible|certified)', 'features'),
        ]
    
    def detect_technical_question(self, query: str) -> Optional[str]:
        """
        Detect if query is asking a technical question.
        
        Args:
            query: User's question
            
        Returns:
            Question type if detected, None otherwise
        """
        query_lower = query.lower()
        
        for pattern, question_type in self.question_patterns:
            if re.search(pattern, query_lower):
                return question_type
        
        return None
    
    def answer_technical_question(
        self,
        query: str,
        products: List[Product],
        question_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Answer a technical question about products in context.
        
        Args:
            query: User's question
            products: Products currently in context
            question_type: Optional pre-detected question type
            
        Returns:
            Answer string if question can be answered, None otherwise
        """
        if not products:
            return None
        
        # Detect question type if not provided
        if question_type is None:
            question_type = self.detect_technical_question(query)
        
        if question_type is None:
            return None
        
        # Route to appropriate handler
        handler_map = {
            'data_rate': self._answer_data_rate,
            'resolution': self._answer_resolution,
            'length': self._answer_length,
            'connectors': self._answer_connectors,
            'features': self._answer_features,
            'active_passive': self._answer_active_passive,
            'shielded': self._answer_shielded,
            'color': self._answer_color,
            'wire_gauge': self._answer_wire_gauge,
            'plating': self._answer_plating,
            'resolution_detail': self._answer_resolution_detail,
            'warranty': self._answer_warranty,
        }

        handler = handler_map.get(question_type)
        if handler:
            return handler(query, products)

        return None
    
    def _answer_data_rate(self, query: str, products: List[Product]) -> str:
        """Answer data rate / bandwidth questions."""
        
        # Check connector types to determine bandwidth
        all_high_speed = all(
            any('high-speed' in str(f).lower() or 'high speed' in str(f).lower() 
                for f in p.metadata.get('features', []))
            for p in products
        )
        
        has_hdmi = any(
            any('hdmi' in str(c).lower() for c in p.metadata.get('connectors', []))
            for p in products
        )
        
        if has_hdmi and all_high_speed:
            return (
                "These are High-Speed HDMI cables rated for **10.2 Gbps** bandwidth. "
                "That's plenty for 4K streaming at 30Hz and Full HD at 60Hz. "
                "For 4K at 60Hz or higher, you'd want HDMI 2.0 cables (18 Gbps) or HDMI 2.1 (48 Gbps)."
            )
        elif has_hdmi:
            return (
                "These HDMI cables support standard bandwidth rates, typically up to **10.2 Gbps**. "
                "This handles Full HD (1080p) at 60Hz and 4K at 30Hz."
            )
        
        # Fallback
        return "The bandwidth depends on the specific cable type. High-Speed HDMI cables typically support 10.2 Gbps."
    
    def _answer_resolution(self, query: str, products: List[Product]) -> str:
        """Answer resolution support questions."""
        
        # Collect all resolutions from products using unified methods
        resolutions = set()
        for p in products:
            if p.supports_4k():
                resolutions.add('4K at 30Hz')
            if p.supports_resolution('8k'):
                resolutions.add('8K')
            if p.supports_resolution('1080p'):
                resolutions.add('1080p')
        
        if '4K at 30Hz' in resolutions:
            # Check if user asked specifically about 4K
            if '4k' in query.lower():
                return (
                    "Yes, all three support **4K resolution at 30Hz**. This is perfect for streaming services, "
                    "movies, and general use. For 4K gaming at 60Hz or higher, you'd need HDMI 2.0 or newer cables."
                )
            else:
                res_list = ', '.join(sorted(resolutions))
                return f"These cables support: **{res_list}**."
        elif resolutions:
            res_list = ', '.join(sorted(resolutions))
            return f"These cables support: **{res_list}**."
        
        return "The maximum resolution depends on the specific cable. Check the product specifications for details."
    
    def _answer_length(self, query: str, products: List[Product]) -> str:
        """Answer length questions."""
        
        lengths = set()
        for p in products:
            length_display = p.metadata.get('length_display')
            if length_display:
                lengths.add(length_display)
        
        if len(lengths) == 1:
            # All same length
            length = lengths.pop()
            return f"All three are **{length}**."
        elif lengths:
            # Different lengths
            length_list = ', '.join(sorted(lengths))
            return f"The lengths are: **{length_list}**."
        
        return "Length information is available in the product details above."
    
    def _answer_connectors(self, query: str, products: List[Product]) -> str:
        """Answer connector type questions."""
        
        connector_pairs = set()
        for p in products:
            connectors = p.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                # Simplify connector names
                source = self._simplify_connector(connectors[0])
                target = self._simplify_connector(connectors[1])
                connector_pairs.add(f"{source} to {target}")
        
        if len(connector_pairs) == 1:
            pair = connector_pairs.pop()
            return f"All three are **{pair}** cables."
        elif connector_pairs:
            pairs_list = ', '.join(sorted(connector_pairs))
            return f"The connector types are: **{pairs_list}**."
        
        return "Connector information is available in the product details above."
    
    def _simplify_connector(self, connector: str) -> str:
        """Simplify connector name for display."""
        import re
        # Remove quantity prefix
        cleaned = re.sub(r'^\d+\s*x\s*', '', str(connector), flags=re.IGNORECASE).strip()
        cleaned_lower = cleaned.lower()
        
        if 'usb-c' in cleaned_lower or 'type-c' in cleaned_lower:
            return "USB-C"
        elif 'hdmi' in cleaned_lower:
            return "HDMI"
        elif 'displayport' in cleaned_lower:
            return "DisplayPort"
        elif 'thunderbolt' in cleaned_lower:
            return "Thunderbolt"
        else:
            # Remove parentheses content
            cleaned = re.sub(r'\([^)]*\)', '', cleaned).strip()
            return cleaned if cleaned else str(connector)
    
    def _answer_features(self, query: str, products: List[Product]) -> str:
        """Answer feature support questions."""

        query_lower = query.lower()

        # Detect what feature they're asking about
        if '4k' in query_lower:
            return self._check_feature(products, '4K', "4K at 30Hz")
        elif 'hdr' in query_lower:
            return self._check_feature(products, 'HDR', "HDR")
        elif 'power delivery' in query_lower or 'pd' in query_lower:
            return self._check_feature(products, 'Power Delivery', "Power Delivery")
        elif 'thunderbolt' in query_lower:
            return self._check_feature(products, 'Thunderbolt', "Thunderbolt")
        elif any(term in query_lower for term in ['audio', 'headphone', 'head phone', '3.5mm', 'jack', 'sound']):
            return self._check_feature(products, 'Audio', "audio/headphone jack")
        elif 'ethernet' in query_lower or 'network' in query_lower or 'lan' in query_lower:
            return self._check_feature(products, 'Gigabit', "Gigabit Ethernet")

        # General features list
        all_features = set()
        for p in products:
            features = p.metadata.get('features', [])
            all_features.update(features)

        if all_features:
            features_list = ', '.join(sorted(all_features))
            return f"The features across these products include: **{features_list}**."

        return "Feature information is available in the product details above."
    
    def _check_feature(self, products: List[Product], feature: str, display_name: str) -> str:
        """Check if all, some, or none of the products have a feature."""

        # Use unified methods for resolution features for consistency
        feature_lower = feature.lower()
        if feature_lower in ('4k', '8k', '1080p', '1440p'):
            count = sum(1 for p in products if p.supports_resolution(feature_lower))
        else:
            count = sum(
                1 for p in products
                if any(feature.lower() in f.lower() for f in p.metadata.get('features', []))
            )
        
        total = len(products)
        
        if count == total:
            return f"Yes, all three support **{display_name}**."
        elif count > 0:
            return f"**{count} out of {total}** support {display_name}."
        else:
            return f"None of these products support {display_name}."
    
    def _answer_active_passive(self, query: str, products: List[Product]) -> str:
        """Answer active vs passive cable questions."""
        
        active_count = sum(
            1 for p in products
            if any('active' in str(f).lower() for f in p.metadata.get('features', []))
        )
        
        total = len(products)
        
        if active_count == total:
            return (
                "All three are **active cables** with built-in signal amplification. "
                "This ensures reliable signal quality over longer distances."
            )
        elif active_count > 0:
            return (
                f"**{active_count} out of {total}** are active cables with built-in signal amplification. "
                f"The others are passive cables."
            )
        else:
            return (
                "These are **passive cables**. For distances over 25 feet or challenging installations, "
                "you might want to consider active cables with built-in signal amplification."
            )
    
    def _answer_shielded(self, query: str, products: List[Product]) -> str:
        """Answer shielding questions."""
        
        shielded_count = sum(
            1 for p in products
            if 'Shielded' in p.metadata.get('features', [])
        )
        
        total = len(products)
        
        if shielded_count == total:
            return (
                "Yes, all three feature **shielded construction** to reduce interference "
                "and ensure reliable signal quality."
            )
        elif shielded_count > 0:
            return f"**{shielded_count} out of {total}** have shielded construction."
        else:
            return "These cables use standard construction without additional shielding."
    
    def _answer_color(self, query: str, products: List[Product]) -> str:
        """Answer color questions using extended specs."""

        # First try extended specs
        colors = set()
        for p in products:
            color = p.metadata.get('color')
            if color:
                colors.add(color)

        # Fallback to product name if no extended specs
        if not colors:
            for p in products:
                name = p.metadata.get('name', '').lower()
                if 'black' in name:
                    colors.add('Black')
                elif 'white' in name:
                    colors.add('White')
                elif 'gray' in name or 'grey' in name:
                    colors.add('Gray')

        if len(colors) == 1:
            color = colors.pop()
            return f"All of these are **{color}**."
        elif colors:
            colors_list = ', '.join(sorted(colors))
            return f"The colors are: **{colors_list}**."

        return "Most of our cables come in standard black. Check the product names for specific color options."

    def _answer_wire_gauge(self, query: str, products: List[Product]) -> str:
        """Answer wire gauge / thickness questions using extended specs."""

        gauges = {}
        for p in products:
            gauge = p.metadata.get('wire_gauge')
            if gauge:
                gauges[p.product_number] = gauge

        if not gauges:
            return "Wire gauge information isn't available for these products."

        # Check if asking about best for long distance
        query_lower = query.lower()
        if 'long' in query_lower or 'distance' in query_lower or 'run' in query_lower:
            # Lower AWG = thicker = better for long runs
            # Parse AWG numbers and find lowest
            awg_values = []
            for sku, gauge in gauges.items():
                import re
                match = re.search(r'(\d+)\s*AWG', gauge)
                if match:
                    awg_values.append((int(match.group(1)), sku, gauge))

            if awg_values:
                awg_values.sort()  # Sort by AWG number (lowest first = thickest)
                best_awg, best_sku, best_gauge = awg_values[0]
                return (
                    f"For long cable runs, thicker wire (lower AWG) is better. "
                    f"**{best_sku}** has the thickest wire at **{best_gauge}**. "
                    f"Lower AWG = thicker wire = better signal over distance."
                )

        # General gauge info
        if len(set(gauges.values())) == 1:
            gauge = list(gauges.values())[0]
            return f"All of these use **{gauge}** wire."
        else:
            gauge_list = [f"{sku}: {gauge}" for sku, gauge in gauges.items()]
            return f"Wire gauges vary:\n" + "\n".join(f"- {g}" for g in gauge_list)

    def _answer_plating(self, query: str, products: List[Product]) -> str:
        """Answer connector plating / gold-plated questions using extended specs."""

        plating_info = {}
        for p in products:
            plating = p.metadata.get('connector_plating')
            if plating:
                plating_info[p.product_number] = plating

        if not plating_info:
            return "Connector plating information isn't available for these products."

        gold_plated = [sku for sku, plating in plating_info.items() if 'gold' in plating.lower()]

        if len(gold_plated) == len(products):
            return "Yes, **all of these have gold-plated connectors** for better conductivity and corrosion resistance."
        elif gold_plated:
            return f"**{len(gold_plated)} out of {len(products)}** have gold-plated connectors: {', '.join(gold_plated)}"
        else:
            platings = list(set(plating_info.values()))
            return f"These cables have {', '.join(platings)} connectors (not gold-plated)."

    def _answer_resolution_detail(self, query: str, products: List[Product]) -> str:
        """Answer detailed resolution / refresh rate questions using extended specs."""

        res_details = {}
        for p in products:
            res = p.metadata.get('max_resolution_detail')
            if res:
                res_details[p.product_number] = res

        if not res_details:
            # Fall back to features
            return self._answer_resolution(query, products)

        query_lower = query.lower()

        # Check for specific Hz questions
        if '60' in query_lower:
            supports_60hz = [sku for sku, res in res_details.items() if '60' in res]
            if supports_60hz:
                return f"**{len(supports_60hz)} out of {len(products)}** support 4K @ 60Hz: {', '.join(supports_60hz)}"
            else:
                return "None of these explicitly support 4K @ 60Hz. Most support 4K @ 30Hz."
        elif '30' in query_lower:
            supports_30hz = [sku for sku, res in res_details.items() if '30' in res]
            if supports_30hz:
                return f"**{len(supports_30hz)}** support 4K @ 30Hz: {', '.join(supports_30hz)}"
            return "Check the detailed specs for refresh rate support."

        # General resolution detail
        if len(set(res_details.values())) == 1:
            res = list(res_details.values())[0]
            return f"All of these support **{res}**."
        else:
            res_list = [f"- {sku}: {res}" for sku, res in res_details.items()]
            return "Resolution support varies:\n" + "\n".join(res_list)

    def _answer_warranty(self, query: str, products: List[Product]) -> str:
        """Answer warranty questions."""

        warranties = {}
        for p in products:
            warranty = p.metadata.get('warranty')
            if warranty:
                warranties[p.product_number] = warranty

        if not warranties:
            return "Warranty information isn't available for these products. Please check the product page or contact StarTech.com support."

        # Check if all have same warranty
        unique_warranties = set(warranties.values())
        if len(unique_warranties) == 1:
            warranty = list(unique_warranties)[0]
            if len(products) == 1:
                sku = products[0].product_number
                return f"Yes, **{sku}** comes with a **{warranty} warranty**."
            else:
                return f"All of these come with a **{warranty} warranty**."
        else:
            # Different warranties
            warranty_list = [f"- {sku}: {warranty}" for sku, warranty in warranties.items()]
            return "Warranty coverage varies:\n" + "\n".join(warranty_list)


# Example usage
if __name__ == "__main__":
    from core.context import Product
    
    # Mock products
    products = [
        Product(
            product_number="HDMIMM15FL",
            content="HDMI Cable - 15ft",
            metadata={
                'name': "15ft Flat High Speed HDMI Cable",
                'length_ft': 15.0,
                'length_display': '15.0 ft [4.6 m]',
                'features': ['4K', 'Shielded'],
                'connectors': ['1 x HDMI (19 pin)', '1 x HDMI (19 pin)']
            },
            score=1.0
        )
    ]
    
    handler = TechnicalQuestionHandler()
    
    # Test questions
    queries = [
        "What's the max data rate transfer?",
        "Do they support 4K?",
        "How long are they?",
        "Are they active or passive cables?",
    ]
    
    print("Technical Question Handler - Test Results")
    print("=" * 60)
    
    for query in queries:
        print(f"\nQ: {query}")
        answer = handler.answer_technical_question(query, products)
        print(f"A: {answer}")