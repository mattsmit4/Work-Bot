"""
Search strategies for ST-Bot.

Implements cascading search with progressive filter relaxation:
- Tier 1: Strict search with all filters
- Tier 2: Relaxed search (drop optional filters like length)
- Tier 3: Broad search (category only)

Includes deduplication, result ranking, and product validation
(filtering out couplers/gender changers from cable searches).
"""

from typing import Optional
from dataclasses import dataclass
from core.context import SearchFilters, SearchResult, Product, DroppedFilter, LengthPreference
from core.product_validator import is_actual_cable
from core.structured_logging import get_logger

# Module-level logger
_logger = get_logger("core.search")


@dataclass
class SearchConfig:
    """
    Configuration for search behavior.
    
    Attributes:
        tier1_min_results: Minimum results to accept Tier 1
        tier2_min_results: Minimum results to accept Tier 2
        max_results: Maximum results to return
        enable_deduplication: Remove duplicate products
    """
    tier1_min_results: int = 1
    tier2_min_results: int = 1
    max_results: int = 10
    enable_deduplication: bool = True


class SearchStrategy:
    """
    Implements cascading search strategy.
    
    Uses progressive filter relaxation to find products:
    1. Tier 1 (Strict): Apply all filters
    2. Tier 2 (Relaxed): Drop optional filters (length, features)
    3. Tier 3 (Broad): Category only
    
    Example:
        strategy = SearchStrategy()
        filters = SearchFilters(
            length=6.0,
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables"
        )
        
        result = strategy.search(filters, search_func=pinecone_search)
        # Returns: SearchResult with products, tier used, filters applied
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize search strategy.
        
        Args:
            config: Search configuration (uses defaults if None)
        """
        self.config = config or SearchConfig()
    
    def search(
        self,
        filters: SearchFilters,
        search_func: callable,
        available_lengths: Optional[list[float]] = None,
    ) -> SearchResult:
        """
        Execute cascading search with progressive filter relaxation.

        Args:
            filters: Extracted search filters
            search_func: Function to call for actual search
                         Signature: search_func(filters_dict) -> list[Product]
            available_lengths: Optional list of available lengths in meters
                              for this product type (used for transparency)

        Returns:
            SearchResult with products, tier, and filters used

        Example:
            >>> def mock_search(filters):
            ...     return [Product(...), Product(...)]
            >>>
            >>> result = strategy.search(filters, mock_search)
            >>> print(f"Found {len(result.products)} products using {result.tier}")
        """
        # Store original filters for transparency
        original_filters = self._build_tier1_filters(filters)

        _logger.debug(
            "Starting cascading search",
            extra={
                "event": "search_start",
                "original_filters": original_filters,
            }
        )

        # Try Tier 1: Strict search (all filters)
        tier1_filters = self._build_tier1_filters(filters)
        tier1_products = search_func(tier1_filters)
        # Filter out invalid products (couplers in cable searches)
        tier1_products = self._filter_invalid_products(tier1_products, filters)

        _logger.debug(
            f"Tier 1 search: {len(tier1_products)} products",
            extra={
                "event": "search_tier1",
                "products_found": len(tier1_products),
                "filters": tier1_filters,
            }
        )

        if len(tier1_products) >= self.config.tier1_min_results:
            products = self._deduplicate(tier1_products) if self.config.enable_deduplication else tier1_products
            products = self._rank_and_limit(products, filters)
            return SearchResult(
                products=products,
                filters_used=tier1_filters,
                tier="tier1",
                total_count=len(tier1_products),
                original_filters=original_filters,
                dropped_filters=[]  # No filters dropped in tier 1
            )

        # Try Tier 2: Relaxed search (drop optional filters)
        tier2_filters = self._build_tier2_filters(filters)
        dropped_filters = []

        if tier2_filters != tier1_filters:  # Only try if different from Tier 1
            tier2_products = search_func(tier2_filters)
            # Filter out invalid products (couplers in cable searches)
            tier2_products = self._filter_invalid_products(tier2_products, filters)

            if len(tier2_products) >= self.config.tier2_min_results:
                products = self._deduplicate(tier2_products) if self.config.enable_deduplication else tier2_products

                # Track what was dropped
                dropped_filters = self._identify_dropped_filters(
                    filters, tier1_filters, tier2_filters, available_lengths
                )

                # Rank by length preference (closest-up by default)
                products = self._rank_by_length_preference(products, filters)
                # Filter out products with wildly different lengths (e.g., 0.3ft when asking for 6ft)
                products = self._filter_unreasonable_lengths(products, filters)
                products = self._rank_and_limit(products, filters)

                return SearchResult(
                    products=products,
                    filters_used=tier2_filters,
                    tier="tier2",
                    total_count=len(tier2_products),
                    original_filters=original_filters,
                    dropped_filters=dropped_filters
                )

        # Try Tier 2.5: Relax category (cable→adapter) but keep connectors
        # This handles cases like "HDMI to DisplayPort cable" where the product
        # is actually an adapter but user said "cable"
        tier2_5_filters = self._build_tier2_5_filters(filters)
        if tier2_5_filters != tier2_filters:
            tier2_5_products = search_func(tier2_5_filters)
            # Pass actual category to avoid applying cable validation to adapters
            actual_cat = tier2_5_filters.get('category', '')
            tier2_5_products = self._filter_invalid_products(tier2_5_products, filters, actual_cat)

            if len(tier2_5_products) >= self.config.tier2_min_results:
                products = self._deduplicate(tier2_5_products) if self.config.enable_deduplication else tier2_5_products

                dropped_filters = self._identify_dropped_filters(
                    filters, tier1_filters, tier2_5_filters, available_lengths
                )

                products = self._rank_by_length_preference(products, filters)
                # Filter out products with wildly different lengths
                products = self._filter_unreasonable_lengths(products, filters)
                products = self._rank_and_limit(products, filters)

                return SearchResult(
                    products=products,
                    filters_used=tier2_5_filters,
                    tier="tier2.5",
                    total_count=len(tier2_5_products),
                    original_filters=original_filters,
                    dropped_filters=dropped_filters,
                    category_relaxed=True  # Flag that we relaxed cable→adapter
                )

        # Try Tier 3: Keep connectors, drop category entirely
        # This is safer than old tier3 which dropped connectors
        # EXCEPT for docks/hubs: they don't have connector metadata, so dropping
        # category would match unrelated products (e.g., USB-C cables instead of docks)
        is_dock_or_hub_search = filters.product_category and filters.product_category.lower() in (
            'dock', 'docks', 'hub', 'hubs', 'docking station', 'docking stations'
        )

        # Skip tier 3 for dock/hub searches - go straight to tier 4 which keeps category
        if not is_dock_or_hub_search:
            tier3_filters = self._build_tier3_filters(filters)
            tier3_products = search_func(tier3_filters)
            tier3_products = self._filter_invalid_products(tier3_products, filters)

            if len(tier3_products) >= self.config.tier2_min_results:
                dropped_filters = self._identify_dropped_filters(
                    filters, tier1_filters, tier3_filters, available_lengths
                )

                products = self._deduplicate(tier3_products) if self.config.enable_deduplication else tier3_products
                products = self._rank_by_length_preference(products, filters)
                # Filter out products with wildly different lengths
                products = self._filter_unreasonable_lengths(products, filters)
                products = self._rank_and_limit(products, filters)

                return SearchResult(
                    products=products,
                    filters_used=tier3_filters,
                    tier="tier3",
                    total_count=len(tier3_products),
                    original_filters=original_filters,
                    dropped_filters=dropped_filters
                )

        # Tier 4 (last resort): Category only, no connectors
        # Only use this if we truly have nothing
        tier4_filters = self._build_tier4_filters(filters)
        tier4_products = search_func(tier4_filters)
        tier4_products = self._filter_invalid_products(tier4_products, filters)

        dropped_filters = self._identify_dropped_filters(
            filters, tier1_filters, tier4_filters, available_lengths
        )

        products = self._deduplicate(tier4_products) if self.config.enable_deduplication else tier4_products
        products = self._rank_by_length_preference(products, filters)
        # Filter out products with wildly different lengths
        products = self._filter_unreasonable_lengths(products, filters)
        products = self._rank_and_limit(products, filters)

        return SearchResult(
            products=products,
            filters_used=tier4_filters,
            tier="tier4",
            total_count=len(tier4_products),
            original_filters=original_filters,
            dropped_filters=dropped_filters
        )
    
    # === Filter Building Methods ===
    
    def _build_tier1_filters(self, filters: SearchFilters) -> dict:
        """
        Build Tier 1 filters (strict - all filters applied).

        Args:
            filters: Extracted search filters

        Returns:
            Dictionary of filters for search
        """
        filter_dict = {}

        # Category
        if filters.product_category:
            filter_dict['category'] = filters.product_category

        # Connectors
        if filters.connector_from:
            filter_dict['connector_from'] = filters.connector_from
        if filters.connector_to:
            filter_dict['connector_to'] = filters.connector_to
            # Flag same-connector cables (HDMI-to-HDMI, USB-C-to-USB-C)
            if filters.connector_to == filters.connector_from:
                filter_dict['same_connector'] = True

        # Length (optional but included in Tier 1)
        if filters.length and filters.length_unit:
            filter_dict['length'] = filters.length
            filter_dict['length_unit'] = filters.length_unit
            filter_dict['length_preference'] = filters.length_preference

        # Features
        if filters.features:
            filter_dict['features'] = filters.features

        # Port count (for hubs, switches)
        if filters.port_count:
            filter_dict['port_count'] = filters.port_count

        # Color (optional)
        if filters.color:
            filter_dict['color'] = filters.color

        # Keywords for text matching (critical for non-cable products)
        if filters.keywords:
            filter_dict['keywords'] = filters.keywords

        return filter_dict

    def _build_tier2_filters(self, filters: SearchFilters) -> dict:
        """
        Build Tier 2 filters (relaxed - drop optional filters).

        Drops:
        - Length requirements
        - Feature requirements

        Keeps:
        - Category
        - Connectors
        - Keywords (essential for text matching)

        Args:
            filters: Extracted search filters

        Returns:
            Dictionary of filters for search
        """
        filter_dict = {}

        # Category
        if filters.product_category:
            filter_dict['category'] = filters.product_category

        # Connectors (keep these - they're important)
        if filters.connector_from:
            filter_dict['connector_from'] = filters.connector_from
        if filters.connector_to and filters.connector_to != filters.connector_from:
            filter_dict['connector_to'] = filters.connector_to

        # Port count (keep - important for hub/switch searches)
        if filters.port_count:
            filter_dict['port_count'] = filters.port_count

        # Keywords (keep - essential for non-cable products)
        if filters.keywords:
            filter_dict['keywords'] = filters.keywords

        # Drop: length, features

        return filter_dict

    def _build_tier2_5_filters(self, filters: SearchFilters) -> dict:
        """
        Build Tier 2.5 filters (relax category, keep connectors).

        When user asks for "HDMI to DisplayPort cable" but products are
        categorized as "adapter", this tier tries the alternate category.

        Args:
            filters: Extracted search filters

        Returns:
            Dictionary of filters for search
        """
        filter_dict = {}

        # Swap category: cables→adapters, adapters→cables
        category_swaps = {
            'cables': 'Adapters',
            'cable': 'Adapters',
            'adapters': 'Cables',
            'adapter': 'Cables',
        }

        original_category = (filters.product_category or '').lower()
        if original_category in category_swaps:
            filter_dict['category'] = category_swaps[original_category]
        else:
            # No swap available, return same as tier2
            return self._build_tier2_filters(filters)

        # Keep connectors (these are critical)
        if filters.connector_from:
            filter_dict['connector_from'] = filters.connector_from
        if filters.connector_to and filters.connector_to != filters.connector_from:
            filter_dict['connector_to'] = filters.connector_to

        # Keywords (keep - essential for non-cable products)
        if filters.keywords:
            filter_dict['keywords'] = filters.keywords

        return filter_dict

    def _build_tier3_filters(self, filters: SearchFilters) -> dict:
        """
        Build Tier 3 filters (drop category, keep connectors and keywords).

        This is safer than dropping connectors - connectors are the key
        product differentiator. Keywords are also kept for text matching.

        Args:
            filters: Extracted search filters

        Returns:
            Dictionary of filters for search
        """
        filter_dict = {}

        # Keep connectors but drop category
        if filters.connector_from:
            filter_dict['connector_from'] = filters.connector_from
        if filters.connector_to and filters.connector_to != filters.connector_from:
            filter_dict['connector_to'] = filters.connector_to

        # Keywords (keep - essential for non-cable products)
        if filters.keywords:
            filter_dict['keywords'] = filters.keywords

        # If no connectors AND no keywords, fall back to tier4 logic
        if not filter_dict.get('connector_from') and not filter_dict.get('keywords'):
            return self._build_tier4_filters(filters)

        return filter_dict

    def _build_tier4_filters(self, filters: SearchFilters) -> dict:
        """
        Build Tier 4 filters (last resort - category and keywords only).

        Only used when all other tiers fail. Keywords are still kept to
        ensure relevant text matching.

        Args:
            filters: Extracted search filters

        Returns:
            Dictionary of filters for search
        """
        filter_dict = {}

        # Category (or default to "Cables" if nothing specified)
        if filters.product_category:
            filter_dict['category'] = filters.product_category
        else:
            # Default fallback
            filter_dict['category'] = 'Cables'

        # Keywords (keep even in tier 4 - essential for relevance)
        if filters.keywords:
            filter_dict['keywords'] = filters.keywords

        return filter_dict
    
    # === Result Processing Methods ===
    
    def _deduplicate(self, products: list[Product]) -> list[Product]:
        """
        Remove duplicate products based on product_number.

        Also treats marketplace variants (e.g., -VAMZ suffix for Amazon) as
        duplicates of the base product to avoid showing essentially identical
        products.

        Args:
            products: List of products (may contain duplicates)

        Returns:
            List of unique products (first occurrence kept)
        """
        seen = set()
        unique_products = []

        for product in products:
            # Get base SKU by stripping marketplace variant suffixes
            base_sku = self._get_base_sku(product.product_number)

            if base_sku not in seen:
                seen.add(base_sku)
                unique_products.append(product)

        return unique_products

    def _get_base_sku(self, sku: str) -> str:
        """
        Get base SKU by normalizing variants to avoid duplicates.

        Handles:
        - Marketplace variants: -VAMZ (Amazon)
        - Color variants: MBNL (black) vs MWNL (white) at end of SKU

        Args:
            sku: Full product SKU

        Returns:
            Normalized SKU for deduplication

        Examples:
            "CDP2HD2MBNL-VAMZ" -> "CDP2HD2MxNL"
            "CDP2HD2MBNL" -> "CDP2HD2MxNL"
            "CDP2HD2MWNL" -> "CDP2HD2MxNL"
            "CDP2HD1MBNL" -> "CDP2HD1MxNL"
            "CDP2HD1MWNL" -> "CDP2HD1MxNL"
        """
        result = sku

        # Strip marketplace variant suffixes
        variant_suffixes = ['-VAMZ']
        for suffix in variant_suffixes:
            if result.endswith(suffix):
                result = result[:-len(suffix)]

        # Normalize color variants at end of SKU
        # Pattern: ...M[B/W]NL where B=black, W=white
        # Replace with ...MxNL to treat as same product
        import re
        result = re.sub(r'M[BW]NL$', 'MxNL', result)

        return result

    def _filter_invalid_products(
        self,
        products: list[Product],
        filters: SearchFilters,
        actual_category: str = None
    ) -> list[Product]:
        """
        Filter out products that don't match the requested category type.

        When searching for cables, excludes couplers/gender changers that
        are miscategorized in the data (e.g., GCHDMIFF is in "HDMI Cables"
        but is actually a coupler with no length).

        Args:
            products: Raw products from search
            filters: Search filters (used to determine category)
            actual_category: Override category (used when tier 2.5 swaps cable→adapter)

        Returns:
            Filtered list of valid products for the category
        """
        # Only apply cable validation when searching cable categories
        cable_categories = {'cables', 'cable', 'hdmi cables', 'displayport cables',
                           'usb cables', 'digital display cables'}

        # Use actual_category if provided (tier 2.5), otherwise use filters
        category = (actual_category or filters.product_category or '').lower()

        if category in cable_categories:
            # Filter out couplers/gender changers from cable searches
            valid_products = [p for p in products if is_actual_cable(p)]
            return valid_products

        # For non-cable categories (including adapters), return as-is
        return products

    def _rank_and_limit(
        self,
        products: list[Product],
        filters: SearchFilters
    ) -> list[Product]:
        """
        Rank products by relevance and limit to max_results.

        Ranking criteria:
        1. Exact length match (if length specified)
        2. Has all requested features
        3. Similarity score (already in Product.score)

        When user indicates length flexibility (EXACT_OR_SHORTER or CLOSEST),
        ensures variety by including products at different lengths.

        Args:
            products: List of products to rank
            filters: Original search filters (for relevance scoring)

        Returns:
            Ranked and limited list of products
        """
        # Score each product
        scored_products = []
        for product in products:
            relevance_score = self._calculate_relevance(product, filters)
            scored_products.append((relevance_score, product))

        # Sort by relevance (descending) then by original score
        scored_products.sort(key=lambda x: (x[0], x[1].score), reverse=True)

        # Extract just products for further processing
        ranked_products = [product for _, product in scored_products]

        # Apply length variety if user indicated flexibility
        if self._should_diversify_lengths(filters):
            ranked_products = self._diversify_by_length(
                ranked_products, filters, limit=self.config.max_results
            )
        else:
            ranked_products = ranked_products[:self.config.max_results]

        return ranked_products

    def _should_diversify_lengths(self, filters: SearchFilters) -> bool:
        """
        Check if we should diversify results by length.

        Only diversify when user indicated length flexibility AND
        specified a length preference.

        Args:
            filters: Search filters

        Returns:
            True if length diversification should be applied
        """
        if not filters.length:
            return False

        # Diversify when user said "shorter is fine" or wants "closest"
        return filters.length_preference in (
            LengthPreference.EXACT_OR_SHORTER,
            LengthPreference.CLOSEST
        )

    def _diversify_by_length(
        self,
        products: list[Product],
        filters: SearchFilters,
        limit: int
    ) -> list[Product]:
        """
        Select products ensuring variety in cable lengths.

        When user indicates flexibility (e.g., "shorter is fine"), include
        products at different lengths rather than multiple products at the
        same length.

        Strategy:
        1. Always include the best match (closest to requested length)
        2. Include one shorter option if available and user accepts shorter
        3. Include one longer option for comparison
        4. Fill remaining slots by relevance

        Args:
            products: Ranked list of products
            filters: Search filters with length preference
            limit: Maximum number of products to return

        Returns:
            Diversified list of products
        """
        if not products or limit <= 0:
            return []

        if not filters.length or not filters.length_unit:
            return products[:limit]

        requested_m = self._normalize_length(filters.length, filters.length_unit)

        # Categorize products by length relative to request
        shorter = []  # Products shorter than requested
        longer = []  # Products longer than requested (includes "at length")

        for product in products:
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit', 'm')

            if not product_length:
                longer.append(product)  # No length info, put at end
                continue

            product_m = self._normalize_length(float(product_length), product_unit)
            diff = product_m - requested_m

            if diff < -0.05:  # Clearly shorter (more than ~2 inches under)
                shorter.append(product)
            else:
                longer.append(product)  # At or above requested length

        # Sort each category by distance from requested (closest first)
        def distance_key(p):
            pl = p.metadata.get('length')
            if not pl:
                return float('inf')
            pm = self._normalize_length(float(pl), p.metadata.get('length_unit', 'm'))
            return abs(pm - requested_m)

        shorter.sort(key=distance_key)
        longer.sort(key=distance_key)

        # Build diverse result set
        result = []

        # 1. Best match first - closest to requested (usually from longer/at_length)
        if longer:
            result.append(longer[0])
            longer = longer[1:]

        # 2. Add shorter option if user accepts shorter and one exists
        if (shorter and len(result) < limit and
            filters.length_preference in (LengthPreference.EXACT_OR_SHORTER,
                                          LengthPreference.CLOSEST)):
            result.append(shorter[0])
            shorter = shorter[1:]

        # 3. Add another longer option for comparison if we have room
        if longer and len(result) < limit:
            result.append(longer[0])
            longer = longer[1:]

        # 4. Fill remaining slots from remaining products by distance
        remaining = shorter + longer
        remaining.sort(key=distance_key)
        for product in remaining:
            if len(result) >= limit:
                break
            if product not in result:
                result.append(product)

        return result[:limit]
    
    def _calculate_relevance(
        self,
        product: Product,
        filters: SearchFilters
    ) -> float:
        """
        Calculate relevance score for a product based on filters.

        Args:
            product: Product to score
            filters: Search filters

        Returns:
            Relevance score (0.0 - 2.0, where 1.0+ indicates primary products)
        """
        score = 0.0
        checks = 0

        # Check length match (if specified)
        if filters.length and filters.length_unit:
            checks += 1
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit')

            if product_length and product_unit:
                # Convert to same unit for comparison
                filter_length_normalized = self._normalize_length(
                    filters.length, filters.length_unit
                )
                product_length_normalized = self._normalize_length(
                    product_length, product_unit
                )

                # Exact match or within 10% tolerance
                if abs(filter_length_normalized - product_length_normalized) / filter_length_normalized < 0.1:
                    score += 1.0

        # Check features match (if specified)
        if filters.features:
            checks += 1
            product_features = set(product.metadata.get('features', []))
            requested_features = set(filters.features)

            # Score based on how many requested features are present
            if requested_features:
                # Use unified resolution methods for resolution features
                # This ensures consistent 4K/8K/1440p/1080p detection
                resolution_features = {'4K', '8K', '1080p', '1440p'}
                matching_count = 0

                for feature in requested_features:
                    feature_upper = feature.upper() if feature else ''
                    if feature_upper in resolution_features:
                        # Use unified Product method for resolution features
                        if product.supports_resolution(feature.lower()):
                            matching_count += 1
                    elif feature in product_features or feature.lower() in [f.lower() for f in product_features]:
                        # Standard feature matching for non-resolution features
                        matching_count += 1

                score += matching_count / len(requested_features)

        # Category-specific relevance boost
        # Boost "primary" products over accessories when searching for a category
        category_boost = self._calculate_category_boost(product, filters)

        # Return average score (or 0.5 if no checks) plus category boost
        base_score = score / checks if checks > 0 else 0.5
        return base_score + category_boost

    def _calculate_category_boost(
        self,
        product: Product,
        filters: SearchFilters
    ) -> float:
        """
        Calculate relevance boost based on category-specific metadata.

        When a user searches for "server rack", boost actual racks (have UHEIGHT)
        over rack accessories (no UHEIGHT). When searching for "PCI card", boost
        actual cards (have BUSTYPE) over tools/accessories.

        Args:
            product: Product to score
            filters: Search filters

        Returns:
            Boost value (0.0 = accessory, 1.0 = primary product)
        """
        category = (filters.product_category or '').lower()
        metadata = product.metadata
        sub_category = (metadata.get('sub_category', '') or '').lower()

        # Server racks: boost products with BOTH UHEIGHT and RACKTYPE (actual racks)
        # Products with only UHEIGHT (drawers, shelves) get partial boost
        if category in ('server racks', 'racks'):
            has_uheight = metadata.get('UHEIGHT') and str(metadata.get('UHEIGHT')).lower() not in ('nan', '')
            has_racktype = metadata.get('RACKTYPE') and str(metadata.get('RACKTYPE')).lower() not in ('nan', '')

            if has_uheight and has_racktype:
                return 1.0  # This is an actual rack (has both U-height and rack type)
            elif has_uheight:
                return 0.5  # Rack accessory with U-height (drawer, shelf) - partial boost
            elif 'accessories' in sub_category or 'shelves' in sub_category:
                return 0.0  # Pure accessory (cage nuts, screws)
            return 0.3  # Unknown

        # Computer cards: boost products with BUSTYPE containing "PCI"
        if category in ('computer cards', 'cards'):
            bus_type = metadata.get('BUSTYPE', '')
            if bus_type and 'pci' in str(bus_type).lower():
                return 1.0  # This is an actual PCI card
            elif 'accessories' in sub_category or 'tools' in sub_category:
                return 0.0  # This is a tool/accessory
            return 0.3  # Unknown

        # Storage enclosures: boost actual enclosures over accessories
        if category in ('storage enclosures', 'enclosures'):
            drive_size = metadata.get('drive_size') or metadata.get('DRIVESIZE')
            if drive_size and str(drive_size).lower() not in ('nan', ''):
                return 1.0  # This is an actual enclosure
            return 0.3

        # No category-specific boost needed
        return 0.0
    
    def _normalize_length(self, length: float, unit: str) -> float:
        """
        Normalize length to meters for comparison.
        
        Args:
            length: Length value
            unit: Length unit
            
        Returns:
            Length in meters
        """
        if unit == 'm':
            return length
        elif unit == 'ft':
            return length * 0.3048
        elif unit == 'in':
            return length * 0.0254
        elif unit == 'cm':
            return length * 0.01
        else:
            return length


    def _identify_dropped_filters(
        self,
        original: SearchFilters,
        tier1_filters: dict,
        used_filters: dict,
        available_lengths: Optional[list[float]] = None
    ) -> list[DroppedFilter]:
        """
        Identify which filters were dropped between tier1 and used filters.

        Args:
            original: Original SearchFilters object
            tier1_filters: Filters from tier 1 (all filters)
            used_filters: Filters actually used
            available_lengths: Available lengths in meters for alternatives

        Returns:
            List of DroppedFilter objects describing what was relaxed
        """
        dropped = []

        # Check if length was dropped
        if 'length' in tier1_filters and 'length' not in used_filters:
            requested_length = original.length
            requested_unit = original.length_unit or 'ft'

            # Format the requested value
            requested_str = f"{requested_length}{requested_unit}"

            # Format alternatives if available
            alternatives = None
            if available_lengths:
                alternatives = self._format_length_alternatives(available_lengths)

            dropped.append(DroppedFilter(
                filter_name="length",
                requested_value=requested_str,
                reason=f"No exact {requested_str} cables available",
                alternatives=alternatives
            ))

        # Check if features were dropped
        if 'features' in tier1_filters and 'features' not in used_filters:
            dropped.append(DroppedFilter(
                filter_name="features",
                requested_value=original.features,
                reason="No products with all requested features",
                alternatives=None
            ))

        # Check if connectors were dropped
        if 'connector_from' in tier1_filters and 'connector_from' not in used_filters:
            dropped.append(DroppedFilter(
                filter_name="connector_from",
                requested_value=original.connector_from,
                reason="Connector type not available",
                alternatives=None
            ))

        if 'connector_to' in tier1_filters and 'connector_to' not in used_filters:
            dropped.append(DroppedFilter(
                filter_name="connector_to",
                requested_value=original.connector_to,
                reason="Connector type not available",
                alternatives=None
            ))

        # Check if color was dropped
        if 'color' in tier1_filters and 'color' not in used_filters:
            dropped.append(DroppedFilter(
                filter_name="color",
                requested_value=original.color,
                reason=f"No {original.color} products found",
                alternatives=None
            ))

        return dropped

    def _format_length_alternatives(self, lengths_in_meters: list[float]) -> list[str]:
        """
        Format available lengths for display.

        Args:
            lengths_in_meters: List of lengths in meters

        Returns:
            List of formatted strings like "1m (3.3ft)", "2m (6.6ft)"
        """
        formatted = []
        for length_m in sorted(lengths_in_meters):
            length_ft = length_m * 3.28084
            formatted.append(f"{length_m:.0f}m ({length_ft:.1f}ft)")
        return formatted

    def _rank_by_length_preference(
        self,
        products: list[Product],
        filters: SearchFilters
    ) -> list[Product]:
        """
        Rank products by length according to user preference.

        Default behavior (EXACT_OR_LONGER): Products closest to but >= requested length first.
        EXACT_OR_SHORTER: Products closest to but <= requested length first.
        CLOSEST: Products closest in either direction first.

        Args:
            products: List of products to rank
            filters: Search filters with length preference

        Returns:
            Products sorted by length preference
        """
        if not filters.length or not filters.length_unit:
            return products

        # Convert requested length to meters for comparison
        requested_m = self._normalize_length(filters.length, filters.length_unit)

        def length_sort_key(product: Product) -> tuple:
            """
            Generate sort key for a product based on length preference.

            Returns tuple: (priority_group, distance_from_requested)
            - priority_group: 0 = preferred direction, 1 = other direction
            - distance: absolute distance from requested length
            """
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit', 'm')

            if not product_length:
                # Products without length go to the end
                return (2, float('inf'))

            product_m = self._normalize_length(float(product_length), product_unit)
            diff = product_m - requested_m

            preference = filters.length_preference

            if preference == LengthPreference.EXACT_OR_LONGER:
                # Prefer products >= requested length, sorted by smallest excess
                if diff >= -0.05:  # Small tolerance for "exact" match
                    return (0, abs(diff))
                else:
                    return (1, abs(diff))

            elif preference == LengthPreference.EXACT_OR_SHORTER:
                # Prefer products <= requested length, sorted by smallest deficit
                if diff <= 0.05:  # Small tolerance
                    return (0, abs(diff))
                else:
                    return (1, abs(diff))

            else:  # LengthPreference.CLOSEST
                # Just sort by absolute distance
                return (0, abs(diff))

        return sorted(products, key=length_sort_key)

    def _filter_unreasonable_lengths(
        self,
        products: list[Product],
        filters: SearchFilters,
        min_ratio: float = 0.25,
        max_ratio: float = 4.0
    ) -> list[Product]:
        """
        Filter out products with lengths wildly different from requested.

        When user asks for 6ft cable, returning 0.3ft or 1ft cables is unhelpful
        even if they match connectors. This filters products outside a reasonable
        range of the requested length.

        Args:
            products: List of products to filter
            filters: Search filters with length requirement
            min_ratio: Minimum acceptable ratio (0.25 = at least 25% of requested)
            max_ratio: Maximum acceptable ratio (4.0 = at most 400% of requested)

        Returns:
            Products within reasonable length range, or original list if no length filter
        """
        if not filters.length or not filters.length_unit:
            return products

        # Convert requested length to meters for comparison
        requested_m = self._normalize_length(filters.length, filters.length_unit)

        # Calculate acceptable bounds
        min_length_m = requested_m * min_ratio
        max_length_m = requested_m * max_ratio

        filtered = []
        for product in products:
            product_length = product.metadata.get('length')
            product_unit = product.metadata.get('length_unit', 'm')

            if not product_length:
                # Products without length info - include them (might be adapters, etc.)
                filtered.append(product)
                continue

            try:
                product_m = self._normalize_length(float(product_length), product_unit)
            except (ValueError, TypeError):
                filtered.append(product)
                continue

            # Check if within reasonable range
            if min_length_m <= product_m <= max_length_m:
                filtered.append(product)

        # If all products were filtered out, return original list
        # (better to show something than nothing)
        if not filtered:
            return products

        return filtered


class SearchError(Exception):
    """Raised when search fails."""
    pass