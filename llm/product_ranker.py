"""
Product Ranking Module - Semantic Relevance Scoring

Ranks products by how well they match the user's request.
Considers: length accuracy, feature matches, connector precision, etc.

Design:
- Extensible: Easy to add new scoring criteria
- Transparent: Returns scores with explanations
- Configurable: Adjustable weights for different factors
- Diversity-aware: When user indicates length flexibility, ensures variety
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from core.context import Product, LengthPreference


@dataclass
class RankedProduct:
    """
    A product with its relevance score and match explanation.
    
    Attributes:
        product: The actual product
        score: Overall relevance score (0-100)
        match_quality: "perfect", "excellent", "good", "fair"
        match_reasons: List of why it matches (e.g., "Exact 6ft length", "Has 4K support")
        priority_attributes: Attributes user specifically asked for
    """
    product: Product
    score: float
    match_quality: str
    match_reasons: List[str]
    priority_attributes: Dict[str, any]
    
    def __post_init__(self):
        if self.match_reasons is None:
            self.match_reasons = []
        if self.priority_attributes is None:
            self.priority_attributes = {}


class ProductRanker:
    """
    Ranks products by semantic relevance to user query.
    
    Example:
        >>> ranker = ProductRanker()
        >>> ranked = ranker.rank(products, query="6ft black 4K HDMI cable")
        >>> top3 = ranked[:3]
        >>> print(top3[0].match_quality)  # "perfect"
    """
    
    def __init__(self):
        """Initialize with default scoring weights."""
        
        # Scoring weights (how important each factor is)
        self.weights = {
            'length_match': 30,      # Very important if mentioned
            'feature_match': 25,     # Critical if specific features requested
            'connector_match': 20,   # Important for compatibility
            'attribute_match': 15,   # Color, shielding, etc
            'tier_bonus': 10         # Tier 1 results slightly preferred
        }
    
    def rank(
        self, 
        products: List[Product], 
        query: str,
        extracted_filters: Optional[Dict] = None
    ) -> List[RankedProduct]:
        """
        Rank products by relevance to query.
        
        Args:
            products: List of products to rank
            query: Original user query
            extracted_filters: Filters extracted from query (length, features, etc)
            
        Returns:
            List of RankedProduct sorted by score (highest first)
        """
        query_lower = query.lower()
        ranked = []
        
        for product in products:
            score = 0
            reasons = []
            priority_attrs = {}
            
            # 1. Length scoring (if length was mentioned)
            if extracted_filters and extracted_filters.get('length'):
                requested_length = extracted_filters['length']
                product_length = product.metadata.get('length_ft')
                
                if product_length:
                    length_diff = abs(product_length - requested_length)
                    
                    if length_diff == 0:
                        score += self.weights['length_match']
                        reasons.append(f"Exact {requested_length}ft length - perfect match")
                        priority_attrs['length'] = 'exact'
                    elif length_diff <= 0.5:
                        score += self.weights['length_match'] * 0.95
                        reasons.append(f"Very close to {requested_length}ft at {product_length}ft")
                        priority_attrs['length'] = 'very_close'
                    elif length_diff <= 1:
                        score += self.weights['length_match'] * 0.9
                        if product_length > requested_length:
                            reasons.append(f"Slightly longer at {product_length}ft - gives you extra reach")
                        else:
                            reasons.append(f"Slightly shorter at {product_length}ft - more compact")
                        priority_attrs['length'] = 'close'
                    elif length_diff <= 3:
                        score += self.weights['length_match'] * 0.7
                        if product_length > requested_length:
                            reasons.append(f"Longer at {product_length}ft - good for flexible setups")
                        else:
                            reasons.append(f"Shorter at {product_length}ft - good for tight spaces")
                        priority_attrs['length'] = 'similar'
            else:
                # No length requested - but if user asked for "cable", prefer actual cables over adapters
                product_length = product.metadata.get('length_ft')
                user_wants_cable = 'cable' in query_lower

                if product_length:
                    # Products under 1ft are typically adapters/dongles, not cables
                    is_adapter_length = product_length < 1.0

                    if is_adapter_length and user_wants_cable:
                        # Penalize adapter-length products when user asked for "cable"
                        score -= 15
                        # Don't add a reason - we don't want to highlight this
                    elif product_length >= 3:
                        # Bonus for actual cable lengths when user wants a cable
                        if user_wants_cable:
                            score += 10
                        reasons.append(f"Standard {product_length}ft length")
            
            # 2. Feature scoring (4K, Power Delivery, etc)
            if extracted_filters and extracted_filters.get('features'):
                requested_features = extracted_filters['features']
                product_features = product.metadata.get('features', [])
                
                matches = 0
                product_features_lower = [f.lower() for f in product_features]
                for feature in requested_features:
                    # Case-insensitive feature matching
                    if any(feature.lower() in f for f in product_features_lower):
                        matches += 1
                        reasons.append(f"Has {feature} support")
                        priority_attrs[feature.lower()] = True
                
                if requested_features:
                    feature_ratio = matches / len(requested_features)
                    score += self.weights['feature_match'] * feature_ratio
            
            # 3. Connector precision
            connectors = product.metadata.get('connectors', [])
            if connectors and len(connectors) >= 2:
                source = str(connectors[0]).lower()
                target = str(connectors[1]).lower()
                
                # Check for exact connector mentions in query
                connector_keywords = {
                    'usb-c': ['usb-c', 'usb c', 'type-c'],
                    'hdmi': ['hdmi'],
                    'displayport': ['displayport', 'display port', 'dp'],
                    'thunderbolt': ['thunderbolt', 'tb3', 'tb4'],
                }
                
                for conn_type, keywords in connector_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        if conn_type in source or conn_type in target:
                            score += self.weights['connector_match'] * 0.5
                            priority_attrs['connector'] = conn_type
            
            # 4. Attribute matching (color, shielding, etc)
            # Color matching
            if 'black' in query_lower:
                # We'd need color data in Excel, but for now check product name
                name = product.metadata.get('name', '').lower()
                if 'black' in name or 'blk' in name:
                    score += self.weights['attribute_match']
                    reasons.append("Black color")
                    priority_attrs['color'] = 'black'
            
            # Resolution/bandwidth (if 4K mentioned) - use unified method
            if '4k' in query_lower:
                if product.supports_4k():
                    priority_attrs['resolution'] = '4K'
                    # Reason already added in feature matching
            
            # 5. Hub-specific scoring (port count, USB version, power)
            category = product.metadata.get('category', '').lower()
            if category == 'hub' and 'hub' in query_lower:
                hub_ports = product.metadata.get('hub_ports')
                hub_usb_version = product.metadata.get('hub_usb_version')

                # Heavily penalize products without port count (likely accessories)
                if not hub_ports:
                    score -= 40  # Accessories should rank much lower
                else:
                    # Bonus for actual hubs with port counts
                    score += 20
                    reasons.append(f"{hub_ports}-port hub")

                    # Bonus for USB 3.0+ hubs
                    if hub_usb_version and ('3.0' in hub_usb_version or '3.2' in hub_usb_version):
                        score += 10
                        priority_attrs['usb_version'] = hub_usb_version

                    # Check for port count requests in query
                    import re
                    port_match = re.search(r'(\d+)\s*[-\s]?port', query_lower)
                    if port_match:
                        requested_ports = int(port_match.group(1))
                        if hub_ports == requested_ports:
                            score += 25  # Exact port count match
                            reasons.append(f"Exact {requested_ports}-port match")
                        elif hub_ports >= requested_ports:
                            score += 10  # Has enough ports

            # 5b. Ethernet switch-specific scoring (port count, speed, PoE)
            if category == 'ethernet_switch' and ('switch' in query_lower or 'network' in query_lower):
                hub_ports = product.metadata.get('hub_ports')
                network_speed = product.metadata.get('network_speed', '')
                features = product.metadata.get('features', [])

                # Heavily penalize products without port count
                if not hub_ports:
                    score -= 40
                else:
                    # Bonus for actual switches with port counts
                    score += 20
                    reasons.append(f"{hub_ports}-port switch")

                    # Bonus for Gigabit speed
                    if '1000' in network_speed or 'gigabit' in network_speed.lower():
                        score += 10
                        priority_attrs['speed'] = 'Gigabit'

                    # Bonus for PoE
                    if 'PoE' in features:
                        score += 5
                        priority_attrs['poe'] = True

                    # Check for port count requests in query
                    import re
                    port_match = re.search(r'(\d+)\s*[-\s]?port', query_lower)
                    if port_match:
                        requested_ports = int(port_match.group(1))
                        if hub_ports == requested_ports:
                            score += 30  # Exact port count match (higher weight)
                            reasons.append(f"Exact {requested_ports}-port match")
                        elif hub_ports > requested_ports:
                            score += 15  # Has more ports than requested
                            reasons.append(f"Has {hub_ports} ports (more than requested)")

            # 6. Small tier bonus (tier 1 = strict match is slightly better)
            # This would come from search tier info if available

            # Determine match quality based on score
            if score >= 80:
                quality = "perfect"
            elif score >= 60:
                quality = "excellent"
            elif score >= 40:
                quality = "good"
            else:
                quality = "fair"
            
            ranked.append(RankedProduct(
                product=product,
                score=score,
                match_quality=quality,
                match_reasons=reasons if reasons else ["Meets basic requirements"],
                priority_attributes=priority_attrs
            ))
        
        # Sort by score (highest first)
        ranked.sort(key=lambda x: x.score, reverse=True)

        # Apply diversity if user indicated length flexibility
        length_preference = None
        requested_length = None
        if extracted_filters:
            length_preference = extracted_filters.get('length_preference')
            requested_length = extracted_filters.get('length')

        if self._should_diversify(length_preference, requested_length):
            ranked = self._diversify_by_length(ranked, requested_length, length_preference)

        return ranked

    def _should_diversify(
        self,
        length_preference: Optional[LengthPreference],
        requested_length: Optional[float]
    ) -> bool:
        """
        Check if we should diversify results by length.

        Only diversify when user indicated length flexibility AND
        specified a length preference.
        """
        if not requested_length:
            return False

        return length_preference in (
            LengthPreference.EXACT_OR_SHORTER,
            LengthPreference.CLOSEST
        )

    def _diversify_by_length(
        self,
        ranked: List[RankedProduct],
        requested_length: float,
        length_preference: LengthPreference
    ) -> List[RankedProduct]:
        """
        Ensure variety in results when user indicated length flexibility.

        When user says "shorter is fine" or "around X feet", include products
        at different lengths rather than multiple products at the same length.

        Strategy:
        1. Best match first (closest to requested, or highest score if similar)
        2. Shorter option (if user accepts shorter)
        3. Different length for variety
        """
        if len(ranked) <= 1:
            return ranked

        # Categorize by length relative to requested
        shorter = []  # Clearly shorter
        at_or_above = []  # At or above requested length

        for rp in ranked:
            product_length = rp.product.metadata.get('length_ft')
            if not product_length:
                at_or_above.append(rp)
                continue

            # ~0.5ft tolerance for "at length"
            if product_length < requested_length - 0.5:
                shorter.append(rp)
            else:
                at_or_above.append(rp)

        # Build diverse result - already sorted by score within each category
        result = []

        # 1. Best match first (highest scoring from at_or_above, which is closest)
        if at_or_above:
            result.append(at_or_above[0])
            at_or_above = at_or_above[1:]

        # 2. Shorter option if user accepts shorter
        if shorter and length_preference in (LengthPreference.EXACT_OR_SHORTER,
                                              LengthPreference.CLOSEST):
            result.append(shorter[0])
            shorter = shorter[1:]

        # 3. Fill remaining from at_or_above, then shorter
        remaining = at_or_above + shorter
        for rp in remaining:
            if rp not in result:
                result.append(rp)

        return result
    
    def get_top_n(
        self,
        products: List[Product],
        query: str,
        n: int = 3,
        extracted_filters: Optional[Dict] = None
    ) -> List[RankedProduct]:
        """
        Get top N ranked products.
        
        Args:
            products: Products to rank
            query: User query
            n: Number of products to return (default 3)
            extracted_filters: Optional extracted filters
            
        Returns:
            Top N ranked products
        """
        ranked = self.rank(products, query, extracted_filters)
        return ranked[:n]


# Example usage
if __name__ == "__main__":
    from core.context import Product
    
    # Mock products for testing
    products = [
        Product(
            product_number="CABLE001",
            content="6ft HDMI Cable",
            metadata={
                'name': "6ft HDMI Cable",
                'length_ft': 6.0,
                'features': ['4K', 'HDR'],
                'connectors': ['HDMI', 'HDMI']
            },
            score=1.0
        ),
        Product(
            product_number="CABLE002",
            content="10ft HDMI Cable",
            metadata={
                'name': "10ft HDMI Cable",
                'length_ft': 10.0,
                'features': ['4K'],
                'connectors': ['HDMI', 'HDMI']
            },
            score=1.0
        ),
        Product(
            product_number="CABLE003",
            content="3ft HDMI Cable",
            metadata={
                'name': "3ft HDMI Cable", 
                'length_ft': 3.0,
                'features': ['4K', 'HDR', 'ARC'],
                'connectors': ['HDMI', 'HDMI']
            },
            score=1.0
        ),
    ]
    
    ranker = ProductRanker()
    
    # Test query
    query = "I need a 6ft HDMI cable with 4K support"
    filters = {
        'length': 6.0,
        'features': ['4K']
    }
    
    ranked = ranker.rank(products, query, filters)
    
    print("Ranking Results:")
    print("=" * 60)
    for i, rp in enumerate(ranked, 1):
        print(f"\n{i}. {rp.product.product_number}")
        print(f"   Score: {rp.score:.1f}")
        print(f"   Quality: {rp.match_quality}")
        print(f"   Reasons: {', '.join(rp.match_reasons)}")
        print(f"   Priority: {rp.priority_attributes}")