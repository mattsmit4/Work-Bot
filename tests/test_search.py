"""
Tests for search strategy module.

Run with: pytest tests/test_search.py -v
"""

import pytest
from core.search import SearchStrategy, SearchConfig, SearchError
from core.context import SearchFilters, SearchResult, Product


@pytest.fixture
def strategy():
    """Create a search strategy instance."""
    return SearchStrategy()


@pytest.fixture
def custom_config():
    """Create a custom search config."""
    return SearchConfig(
        tier1_min_results=5,
        tier2_min_results=3,
        max_results=5,
        enable_deduplication=True
    )


@pytest.fixture
def sample_products():
    """Create sample products for testing."""
    return [
        Product("SKU1", "6ft USB-C to HDMI Cable", {"length": 6.0, "length_unit": "ft", "features": ["4K"]}, 0.95),
        Product("SKU2", "10ft USB-C to HDMI Cable", {"length": 10.0, "length_unit": "ft", "features": ["4K"]}, 0.90),
        Product("SKU3", "3ft USB-C to HDMI Cable", {"length": 3.0, "length_unit": "ft"}, 0.85),
        Product("SKU4", "USB-C to HDMI Adapter", {"features": ["4K", "HDCP"]}, 0.80),
        Product("SKU5", "6ft HDMI Cable", {"length": 6.0, "length_unit": "ft"}, 0.75),
    ]


class TestTier1Search:
    """Test Tier 1 (strict) search behavior."""
    
    def test_tier1_with_all_filters(self, strategy, sample_products):
        """Test that Tier 1 applies all filters."""
        filters = SearchFilters(
            length=6.0,
            length_unit="ft",
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables",
            features=["4K"]
        )
        
        # Mock search function that returns products for Tier 1
        def mock_search(filter_dict):
            if filter_dict.get('length') == 6.0:
                return [sample_products[0]]  # 6ft cable
            return []
        
        result = strategy.search(filters, mock_search)
        
        assert result.tier == "tier1"
        assert len(result.products) == 1
        assert result.products[0].product_number == "SKU1"
        assert result.filters_used['length'] == 6.0
        assert result.filters_used['connector_from'] == "USB-C"
    
    def test_tier1_includes_length(self, strategy):
        """Test that Tier 1 includes length filter."""
        filters = SearchFilters(length=6.0, length_unit="ft", product_category="Cables")
        
        def mock_search(filter_dict):
            return []
        
        result = strategy.search(filters, mock_search)
        
        # Even though no results, check Tier 1 tried with length
        tier1_filters = strategy._build_tier1_filters(filters)
        assert 'length' in tier1_filters
        assert tier1_filters['length'] == 6.0
    
    def test_tier1_includes_features(self, strategy):
        """Test that Tier 1 includes feature filters."""
        filters = SearchFilters(
            product_category="Cables",
            features=["4K", "HDCP"]
        )
        
        tier1_filters = strategy._build_tier1_filters(filters)
        assert 'features' in tier1_filters
        assert tier1_filters['features'] == ["4K", "HDCP"]


class TestTier2Search:
    """Test Tier 2 (relaxed) search behavior."""
    
    def test_tier2_when_tier1_fails(self, strategy, sample_products):
        """Test that Tier 2 is used when Tier 1 returns no results."""
        filters = SearchFilters(
            length=15.0,  # No 15ft cables
            length_unit="ft",
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables"
        )
        
        def mock_search(filter_dict):
            if 'length' in filter_dict:
                return []  # Tier 1 fails
            else:
                return sample_products[:3]  # Tier 2 succeeds (no length filter)
        
        result = strategy.search(filters, mock_search)
        
        assert result.tier == "tier2"
        assert len(result.products) > 0
        assert 'length' not in result.filters_used  # Length dropped in Tier 2
        assert 'connector_from' in result.filters_used  # Connectors kept
    
    def test_tier2_drops_length(self, strategy):
        """Test that Tier 2 drops length filter."""
        filters = SearchFilters(
            length=6.0,
            length_unit="ft",
            connector_from="USB-C",
            product_category="Cables"
        )
        
        tier2_filters = strategy._build_tier2_filters(filters)
        assert 'length' not in tier2_filters
        assert 'length_unit' not in tier2_filters
        assert 'connector_from' in tier2_filters
    
    def test_tier2_drops_features(self, strategy):
        """Test that Tier 2 drops feature filters."""
        filters = SearchFilters(
            connector_from="USB-C",
            product_category="Cables",
            features=["4K"]
        )
        
        tier2_filters = strategy._build_tier2_filters(filters)
        assert 'features' not in tier2_filters
        assert 'connector_from' in tier2_filters


class TestTier2_5Search:
    """Test Tier 2.5 (category relaxation) search behavior."""

    def test_tier2_5_swaps_cable_to_adapter(self, strategy):
        """Test that Tier 2.5 swaps cables→adapters."""
        filters = SearchFilters(
            connector_from="HDMI",
            connector_to="DisplayPort",
            product_category="Cables"  # User said "cable"
        )

        tier2_5_filters = strategy._build_tier2_5_filters(filters)
        assert tier2_5_filters['category'] == 'Adapters'  # Swapped to adapter
        assert tier2_5_filters['connector_from'] == 'HDMI'  # Connectors kept
        assert tier2_5_filters['connector_to'] == 'DisplayPort'

    def test_tier2_5_swaps_adapter_to_cable(self, strategy):
        """Test that Tier 2.5 swaps adapters→cables."""
        filters = SearchFilters(
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Adapters"  # User said "adapter"
        )

        tier2_5_filters = strategy._build_tier2_5_filters(filters)
        assert tier2_5_filters['category'] == 'Cables'  # Swapped to cable
        assert tier2_5_filters['connector_from'] == 'USB-C'  # Connectors kept


class TestTier3Search:
    """Test Tier 3 (connectors only) search behavior."""

    def test_tier3_keeps_connectors_drops_category(self, strategy):
        """Test that Tier 3 keeps connectors but drops category."""
        filters = SearchFilters(
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables"
        )

        tier3_filters = strategy._build_tier3_filters(filters)
        assert 'category' not in tier3_filters
        assert tier3_filters['connector_from'] == 'USB-C'
        assert tier3_filters['connector_to'] == 'HDMI'

    def test_tier3_fallback_to_tier4_when_no_connectors(self, strategy):
        """Test that Tier 3 falls back to Tier 4 when no connectors."""
        filters = SearchFilters(
            product_category="Cables"
            # No connectors specified
        )

        tier3_filters = strategy._build_tier3_filters(filters)
        # Should fall back to tier4 logic (category only)
        assert tier3_filters['category'] == 'Cables'


class TestTier4Search:
    """Test Tier 4 (last resort - category only) search behavior."""

    def test_tier4_category_only(self, strategy):
        """Test that Tier 4 only uses category filter."""
        filters = SearchFilters(
            length=6.0,
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables",
            features=["4K"]
        )

        tier4_filters = strategy._build_tier4_filters(filters)
        assert len(tier4_filters) == 1
        assert tier4_filters['category'] == 'Cables'

    def test_tier4_default_category(self, strategy):
        """Test that Tier 4 defaults to Cables if no category."""
        filters = SearchFilters(
            connector_from="USB-C"
        )

        tier4_filters = strategy._build_tier4_filters(filters)
        assert tier4_filters['category'] == 'Cables'


class TestDeduplication:
    """Test product deduplication."""
    
    def test_removes_duplicates(self, strategy):
        """Test that duplicate products are removed."""
        products = [
            Product("SKU1", "Cable 1", {}, 0.9),
            Product("SKU2", "Cable 2", {}, 0.8),
            Product("SKU1", "Cable 1 Duplicate", {}, 0.85),  # Duplicate
            Product("SKU3", "Cable 3", {}, 0.7),
        ]
        
        unique = strategy._deduplicate(products)
        
        assert len(unique) == 3
        assert unique[0].product_number == "SKU1"
        assert unique[1].product_number == "SKU2"
        assert unique[2].product_number == "SKU3"
    
    def test_keeps_first_occurrence(self, strategy):
        """Test that first occurrence is kept when deduplicating."""
        products = [
            Product("SKU1", "First", {}, 0.9),
            Product("SKU1", "Second", {}, 0.95),  # Higher score but duplicate
        ]
        
        unique = strategy._deduplicate(products)
        
        assert len(unique) == 1
        assert unique[0].content == "First"
    
    def test_deduplication_disabled(self, sample_products):
        """Test search with deduplication disabled."""
        config = SearchConfig(enable_deduplication=False)
        strategy = SearchStrategy(config)

        filters = SearchFilters(product_category="Cables")

        # Add duplicates - use only products with length (cables, not adapters)
        # sample_products[3] is an adapter without length, which gets filtered out
        cables_only = [p for p in sample_products if p.metadata.get('length')]
        products_with_dupes = cables_only + [cables_only[0]]

        def mock_search(filter_dict):
            return products_with_dupes

        result = strategy.search(filters, mock_search)

        # Should have duplicates (5 cables + 1 duplicate = 5 after filtering, but no dedup)
        # With deduplication disabled, duplicates are kept
        assert len(result.products) == len(products_with_dupes)


class TestRanking:
    """Test product ranking and limiting."""
    
    def test_limits_results(self, sample_products):
        """Test that results are limited to max_results."""
        config = SearchConfig(max_results=3)
        strategy = SearchStrategy(config)
        
        filters = SearchFilters(product_category="Cables")
        
        def mock_search(filter_dict):
            return sample_products  # 5 products
        
        result = strategy.search(filters, mock_search)
        
        assert len(result.products) == 3
    
    def test_ranks_by_length_match(self, strategy):
        """Test that exact length matches rank higher."""
        products = [
            Product("SKU1", "10ft Cable", {"length": 10.0, "length_unit": "ft"}, 0.8),
            Product("SKU2", "6ft Cable", {"length": 6.0, "length_unit": "ft"}, 0.7),
            Product("SKU3", "3ft Cable", {"length": 3.0, "length_unit": "ft"}, 0.9),
        ]
        
        filters = SearchFilters(length=6.0, length_unit="ft")
        
        # Calculate relevance for each
        relevance1 = strategy._calculate_relevance(products[0], filters)
        relevance2 = strategy._calculate_relevance(products[1], filters)
        relevance3 = strategy._calculate_relevance(products[2], filters)
        
        # 6ft cable should have highest relevance
        assert relevance2 > relevance1
        assert relevance2 > relevance3
    
    def test_ranks_by_feature_match(self, strategy):
        """Test that feature matches rank higher."""
        products = [
            Product("SKU1", "Cable 1", {"features": ["4K", "HDCP"]}, 0.8),
            Product("SKU2", "Cable 2", {"features": ["1080p"]}, 0.9),
            Product("SKU3", "Cable 3", {"features": []}, 0.95),
        ]
        
        filters = SearchFilters(features=["4K", "HDCP"])
        
        relevance1 = strategy._calculate_relevance(products[0], filters)
        relevance2 = strategy._calculate_relevance(products[1], filters)
        relevance3 = strategy._calculate_relevance(products[2], filters)
        
        # Cable with matching features should rank highest
        assert relevance1 > relevance2
        assert relevance1 > relevance3


class TestLengthNormalization:
    """Test length unit normalization."""
    
    def test_normalize_meters(self, strategy):
        """Test meter normalization."""
        assert strategy._normalize_length(2.0, 'm') == 2.0
    
    def test_normalize_feet(self, strategy):
        """Test feet to meter conversion."""
        result = strategy._normalize_length(6.0, 'ft')
        assert abs(result - 1.8288) < 0.001  # 6 ft ≈ 1.83 m
    
    def test_normalize_inches(self, strategy):
        """Test inches to meter conversion."""
        result = strategy._normalize_length(12.0, 'in')
        assert abs(result - 0.3048) < 0.001  # 12 in ≈ 0.30 m
    
    def test_normalize_centimeters(self, strategy):
        """Test centimeter to meter conversion."""
        assert strategy._normalize_length(100.0, 'cm') == 1.0


class TestSearchConfig:
    """Test search configuration."""
    
    def test_custom_config(self):
        """Test custom search configuration."""
        config = SearchConfig(
            tier1_min_results=10,
            tier2_min_results=5,
            max_results=20,
            enable_deduplication=False
        )
        
        assert config.tier1_min_results == 10
        assert config.tier2_min_results == 5
        assert config.max_results == 20
        assert config.enable_deduplication is False
    
    def test_default_config(self):
        """Test default search configuration."""
        config = SearchConfig()
        
        assert config.tier1_min_results == 1
        assert config.tier2_min_results == 1
        assert config.max_results == 10
        assert config.enable_deduplication is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_filters(self, strategy):
        """Test search with empty filters."""
        filters = SearchFilters()

        def mock_search(filter_dict):
            return []

        result = strategy.search(filters, mock_search)

        # Should use Tier 4 (last resort, category only) when no connectors
        assert result.tier == "tier4"
        assert result.filters_used.get('category') == 'Cables'

    def test_no_results_any_tier(self, strategy):
        """Test search when all tiers return no results."""
        filters = SearchFilters(product_category="Cables")

        def mock_search(filter_dict):
            return []  # Always empty

        result = strategy.search(filters, mock_search)

        # Should still return a result (Tier 4, but empty)
        assert result.tier == "tier4"
        assert len(result.products) == 0
    
    def test_only_connectors(self, strategy, sample_products):
        """Test search with only connector filters."""
        filters = SearchFilters(
            connector_from="USB-C",
            connector_to="HDMI"
        )
        
        def mock_search(filter_dict):
            return sample_products[:2]
        
        result = strategy.search(filters, mock_search)
        
        assert len(result.products) > 0
        assert 'connector_from' in result.filters_used


class TestIntegration:
    """Test integration scenarios."""
    
    def test_full_search_flow(self, sample_products):
        """Test complete search flow with all tiers."""
        strategy = SearchStrategy()
        
        # Simulate Tier 1 failing, Tier 2 succeeding
        call_count = [0]
        
        def mock_search(filter_dict):
            call_count[0] += 1
            if 'length' in filter_dict:
                return []  # Tier 1 fails
            elif 'connector_from' in filter_dict:
                return sample_products[:3]  # Tier 2 succeeds
            else:
                return sample_products  # Tier 3
        
        filters = SearchFilters(
            length=20.0,  # No 20ft cables
            length_unit="ft",
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables"
        )
        
        result = strategy.search(filters, mock_search)

        assert result.tier == "tier2"
        # Note: SKU3 (3ft) is filtered out by _filter_unreasonable_lengths
        # because 3ft is below 25% of requested 20ft (min threshold = 5ft)
        assert len(result.products) == 2  # SKU1 (6ft) and SKU2 (10ft)
        assert call_count[0] == 2  # Called Tier 1, then Tier 2
    
    def test_exact_match_tier1(self, sample_products):
        """Test when Tier 1 finds exact match."""
        strategy = SearchStrategy()
        
        filters = SearchFilters(
            length=6.0,
            length_unit="ft",
            connector_from="USB-C",
            connector_to="HDMI",
            product_category="Cables"
        )
        
        def mock_search(filter_dict):
            return [sample_products[0]]  # Perfect match
        
        result = strategy.search(filters, mock_search)
        
        assert result.tier == "tier1"
        assert len(result.products) == 1
        assert result.products[0].product_number == "SKU1"


class TestProductValidation:
    """Test that invalid products (couplers, gender changers) are filtered out."""

    def test_filters_couplers_from_cable_search(self):
        """Test that couplers (no length) are filtered out from cable searches."""
        strategy = SearchStrategy()

        # Mix of real cables and a coupler
        products = [
            Product("SKU1", "6ft HDMI Cable", {"length": 6.0, "length_unit": "ft"}, 0.95),
            Product("GCHDMIFF", "HDMI Coupler", {}, 0.90),  # Coupler - no length
            Product("SKU3", "3ft HDMI Cable", {"length": 3.0, "length_unit": "ft"}, 0.85),
        ]

        filters = SearchFilters(product_category="Cables")

        def mock_search(filter_dict):
            return products

        result = strategy.search(filters, mock_search)

        # Should only have the 2 real cables, not the coupler
        assert len(result.products) == 2
        skus = [p.product_number for p in result.products]
        assert "SKU1" in skus
        assert "SKU3" in skus
        assert "GCHDMIFF" not in skus

    def test_filters_gender_changers_by_sku_prefix(self):
        """Test that products with GC prefix are filtered out."""
        strategy = SearchStrategy()

        products = [
            Product("SKU1", "6ft Cable", {"length": 6.0, "length_unit": "ft"}, 0.95),
            Product("GCHDMI", "HDMI Gender Changer", {"length": 0.5}, 0.90),  # GC prefix
        ]

        filters = SearchFilters(product_category="Cables")

        def mock_search(filter_dict):
            return products

        result = strategy.search(filters, mock_search)

        # GC prefix product should be filtered out
        assert len(result.products) == 1
        assert result.products[0].product_number == "SKU1"

    def test_filters_couplers_by_name_keyword(self):
        """Test that products with 'coupler' in name are filtered out."""
        strategy = SearchStrategy()

        products = [
            Product("SKU1", "6ft Cable", {"length": 6.0, "length_unit": "ft", "name": "6ft HDMI Cable"}, 0.95),
            Product("SKU2", "HDMI F/F", {"length": 0.1, "length_unit": "ft", "name": "HDMI Coupler F/F"}, 0.90),
        ]

        filters = SearchFilters(product_category="Cables")

        def mock_search(filter_dict):
            return products

        result = strategy.search(filters, mock_search)

        # Coupler keyword product should be filtered out
        assert len(result.products) == 1
        assert result.products[0].product_number == "SKU1"

    def test_no_filtering_for_non_cable_categories(self):
        """Test that filtering only applies to cable categories."""
        strategy = SearchStrategy()

        # Adapter without length - should NOT be filtered for "Adapters" category
        products = [
            Product("ADAPT1", "USB-C to HDMI Adapter", {"features": ["4K"]}, 0.95),
        ]

        filters = SearchFilters(product_category="Adapters")

        def mock_search(filter_dict):
            return products

        result = strategy.search(filters, mock_search)

        # Adapter should be kept (not filtered)
        assert len(result.products) == 1
        assert result.products[0].product_number == "ADAPT1"


# Run tests with: pytest tests/test_search.py -v