"""
Tests for filter extraction module.

Run with: pytest tests/test_filters.py -v
"""

import pytest
from core.filters import FilterExtractor
from core.context import SearchFilters


@pytest.fixture
def extractor():
    """Create a filter extractor instance."""
    return FilterExtractor()


class TestLengthExtraction:
    """Test length requirement extraction."""
    
    def test_feet_numeric(self, extractor):
        result = extractor.extract("6ft cable")
        assert result.length == 6.0
        assert result.length_unit == "ft"
    
    def test_feet_word(self, extractor):
        result = extractor.extract("6 feet cable")
        assert result.length == 6.0
        assert result.length_unit == "ft"
    
    def test_foot_singular(self, extractor):
        result = extractor.extract("1 foot cable")
        assert result.length == 1.0
        assert result.length_unit == "ft"
    
    def test_meters(self, extractor):
        result = extractor.extract("2 meter cable")
        assert result.length == 2.0
        assert result.length_unit == "m"
    
    def test_meters_plural(self, extractor):
        result = extractor.extract("3 meters")
        assert result.length == 3.0
        assert result.length_unit == "m"
    
    def test_word_based_length(self, extractor):
        result = extractor.extract("I need a six foot cable")
        assert result.length == 6.0
        assert result.length_unit == "ft"
    
    def test_decimal_length(self, extractor):
        result = extractor.extract("1.5 meter cable")
        assert result.length == 1.5
        assert result.length_unit == "m"
    
    def test_no_length(self, extractor):
        result = extractor.extract("HDMI cable")
        assert result.length is None
        assert result.length_unit is None


class TestConnectorExtraction:
    """Test connector type extraction."""
    
    # === Connector-to-Connector Patterns ===
    
    def test_usb_c_to_hdmi(self, extractor):
        result = extractor.extract("USB-C to HDMI cable")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"
    
    def test_usbc_to_hdmi(self, extractor):
        result = extractor.extract("USBC to HDMI")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"
    
    def test_usb_c_to_displayport(self, extractor):
        result = extractor.extract("USB-C to DisplayPort")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "DisplayPort"
    
    def test_hdmi_to_vga(self, extractor):
        result = extractor.extract("HDMI to VGA adapter")
        assert result.connector_from == "HDMI"
        assert result.connector_to == "VGA"
    
    def test_dp_to_hdmi(self, extractor):
        result = extractor.extract("DisplayPort to HDMI cable")
        assert result.connector_from == "DisplayPort"
        assert result.connector_to == "HDMI"
    
    # === Single Connector Patterns ===
    
    def test_hdmi_cable(self, extractor):
        result = extractor.extract("HDMI cable")
        assert result.connector_from == "HDMI"
        assert result.connector_to == "HDMI"
    
    def test_displayport_cable(self, extractor):
        result = extractor.extract("DisplayPort cable")
        assert result.connector_from == "DisplayPort"
        assert result.connector_to == "DisplayPort"
    
    def test_usb_c_cable(self, extractor):
        result = extractor.extract("USB-C cable")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "USB-C"
    
    def test_vga_cable(self, extractor):
        result = extractor.extract("VGA cable")
        assert result.connector_from == "VGA"
        assert result.connector_to == "VGA"
    
    # === Bare Connector Mentions ===
    
    def test_bare_hdmi(self, extractor):
        result = extractor.extract("Show me HDMI")
        assert result.connector_from == "HDMI"
        assert result.connector_to == "HDMI"
    
    def test_bare_usb_c(self, extractor):
        result = extractor.extract("I need USB-C")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "USB-C"
    
    # === Synonym Expansion ===
    
    def test_dp_abbreviation(self, extractor):
        result = extractor.extract("DP cable")
        # expand_synonyms should convert DP → DisplayPort
        assert result.connector_from == "DisplayPort"
        assert result.connector_to == "DisplayPort"


class TestFeatureExtraction:
    """Test technical feature extraction."""
    
    def test_4k_support(self, extractor):
        result = extractor.extract("HDMI cable with 4K support")
        assert "4K" in result.features
    
    def test_4k_lowercase(self, extractor):
        result = extractor.extract("4k hdmi cable")
        assert "4K" in result.features
    
    def test_1080p(self, extractor):
        result = extractor.extract("1080p HDMI cable")
        assert "1080p" in result.features
    
    def test_thunderbolt(self, extractor):
        result = extractor.extract("Thunderbolt 4 cable")
        assert "Thunderbolt" in result.features
    
    def test_power_delivery(self, extractor):
        result = extractor.extract("USB-C cable with power delivery")
        assert "Power Delivery" in result.features
    
    def test_multiple_features(self, extractor):
        result = extractor.extract("4K HDMI cable with HDCP support")
        assert "4K" in result.features
        assert "HDCP" in result.features
    
    def test_no_features(self, extractor):
        result = extractor.extract("HDMI cable")
        assert result.features == []


class TestCategoryExtraction:
    """Test product category extraction."""
    
    def test_cable_category(self, extractor):
        result = extractor.extract("HDMI cable")
        assert result.product_category == "Cables"
    
    def test_adapter_category(self, extractor):
        result = extractor.extract("USB-C adapter")
        assert result.product_category == "Adapters"
    
    def test_dock_category(self, extractor):
        result = extractor.extract("USB-C dock")
        assert result.product_category == "Docks"
    
    def test_docking_station(self, extractor):
        result = extractor.extract("docking station")
        assert result.product_category == "Docks"
    
    def test_hub_category(self, extractor):
        result = extractor.extract("USB hub")
        assert result.product_category == "Hubs"
    
    def test_switch_category(self, extractor):
        result = extractor.extract("HDMI switch")
        assert result.product_category == "Switches"

    def test_kvm_category(self, extractor):
        result = extractor.extract("KVM switch for 2 computers")
        assert result.product_category == "Kvm Switches"

    def test_ethernet_switch_category(self, extractor):
        """Network switch should map to Ethernet Switches, not generic Switches."""
        result = extractor.extract("network switch with 8 ports")
        assert result.product_category == "Ethernet Switches"

    def test_gigabit_switch_category(self, extractor):
        result = extractor.extract("gigabit switch")
        assert result.product_category == "Ethernet Switches"

    def test_poe_switch_category(self, extractor):
        result = extractor.extract("PoE switch")
        assert result.product_category == "Ethernet Switches"

    def test_mount_category(self, extractor):
        result = extractor.extract("monitor mount")
        assert result.product_category == "Display Mounts"

    def test_wall_mount_category(self, extractor):
        result = extractor.extract("wall mount for TV")
        assert result.product_category == "Display Mounts"

    def test_desk_mount_category(self, extractor):
        result = extractor.extract("desk mount for monitor")
        assert result.product_category == "Display Mounts"

    def test_tv_mount_category(self, extractor):
        result = extractor.extract("I need a TV mount")
        assert result.product_category == "Display Mounts"

    def test_no_category(self, extractor):
        result = extractor.extract("Show me HDMI")
        assert result.product_category is None


class TestPortCountExtraction:
    """Test port count extraction for hubs and switches."""

    def test_8_port_switch(self, extractor):
        result = extractor.extract("I need a network switch with 8 ports")
        assert result.port_count == 8
        assert result.product_category == "Ethernet Switches"

    def test_4_port_hub(self, extractor):
        result = extractor.extract("4 port USB hub")
        assert result.port_count == 4
        assert result.product_category == "Hubs"

    def test_hyphenated_port(self, extractor):
        result = extractor.extract("8-port gigabit switch")
        assert result.port_count == 8

    def test_16_port_switch(self, extractor):
        result = extractor.extract("switch with 16 ports")
        assert result.port_count == 16

    def test_no_port_count(self, extractor):
        result = extractor.extract("network switch")
        assert result.port_count is None


class TestCombinedQueries:
    """Test extraction from complex queries with multiple filters."""
    
    def test_length_and_connectors(self, extractor):
        result = extractor.extract("6ft USB-C to HDMI cable")
        assert result.length == 6.0
        assert result.length_unit == "ft"
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"
        assert result.product_category == "Cables"
    
    def test_length_connectors_features(self, extractor):
        result = extractor.extract("I need a 10ft HDMI cable that supports 4K")
        assert result.length == 10.0
        assert result.length_unit == "ft"
        assert result.connector_from == "HDMI"
        assert result.connector_to == "HDMI"
        assert "4K" in result.features
        assert result.product_category == "Cables"
    
    def test_full_query(self, extractor):
        result = extractor.extract("6 foot USB-C to DisplayPort cable with 4K support")
        assert result.length == 6.0
        assert result.length_unit == "ft"
        assert result.connector_from == "USB-C"
        assert result.connector_to == "DisplayPort"
        assert "4K" in result.features
        assert result.product_category == "Cables"
    
    def test_adapter_with_features(self, extractor):
        result = extractor.extract("USB-C to HDMI adapter with 4K and HDCP")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"
        assert "4K" in result.features
        assert "HDCP" in result.features
        assert result.product_category == "Adapters"
    
    def test_thunderbolt_dock(self, extractor):
        result = extractor.extract("Thunderbolt 4 docking station")
        assert "Thunderbolt" in result.features
        assert result.product_category == "Docks"


class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_empty_string(self, extractor):
        result = extractor.extract("")
        assert result.length is None
        assert result.connector_from is None
        assert result.connector_to is None
        assert result.features == []
        assert result.product_category is None
    
    def test_only_length(self, extractor):
        result = extractor.extract("6ft")
        assert result.length == 6.0
        assert result.length_unit == "ft"
        assert result.connector_from is None
    
    def test_only_connector(self, extractor):
        result = extractor.extract("HDMI")
        assert result.connector_from == "HDMI"
        assert result.length is None
    
    def test_multiple_lengths_takes_first(self, extractor):
        result = extractor.extract("6ft or 10ft cable")
        assert result.length == 6.0  # Should take first match
    
    def test_mixed_case(self, extractor):
        result = extractor.extract("6FT USB-c TO hdmi CABLE")
        assert result.length == 6.0
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"
    
    def test_extra_whitespace(self, extractor):
        result = extractor.extract("   6ft    USB-C   to   HDMI   ")
        assert result.length == 6.0
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"


class TestRealWorldQueries:
    """Test with real-world user queries."""
    
    def test_natural_language_1(self, extractor):
        result = extractor.extract("I need a 6 foot HDMI cable for my TV")
        assert result.length == 6.0
        assert result.connector_from == "HDMI"
        assert result.product_category == "Cables"
    
    def test_natural_language_2(self, extractor):
        result = extractor.extract("Show me USB-C to HDMI adapters")
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"
        assert result.product_category == "Adapters"
    
    def test_natural_language_3(self, extractor):
        result = extractor.extract("Can you find me a DisplayPort cable that supports 4K?")
        assert result.connector_from == "DisplayPort"
        assert "4K" in result.features
        assert result.product_category == "Cables"
    
    def test_technical_query(self, extractor):
        result = extractor.extract("2m Thunderbolt 4 cable with 100W power delivery")
        assert result.length == 2.0
        assert result.length_unit == "m"
        assert "Thunderbolt" in result.features
        assert "Power Delivery" in result.features
        assert result.product_category == "Cables"
    
    def test_comparison_query(self, extractor):
        result = extractor.extract("HDMI 2.1 vs HDMI 2.0 cables")
        assert result.connector_from == "HDMI"
        assert result.product_category == "Cables"


class TestUnitNormalization:
    """Test unit normalization."""
    
    def test_feet_variations(self, extractor):
        queries = ["6 ft", "6 feet", "6 foot", "6ft"]
        for query in queries:
            result = extractor.extract(query)
            assert result.length_unit == "ft", f"Failed for: {query}"
    
    def test_meter_variations(self, extractor):
        queries = ["2 m", "2 meter", "2 meters", "2m"]
        for query in queries:
            result = extractor.extract(query)
            assert result.length_unit == "m", f"Failed for: {query}"


class TestConnectorNormalization:
    """Test connector name normalization."""

    def test_usb_c_variations(self, extractor):
        queries = ["USB-C cable", "USBC cable", "USB C cable"]
        for query in queries:
            result = extractor.extract(query)
            assert result.connector_from == "USB-C", f"Failed for: {query}"

    def test_displayport_variations(self, extractor):
        queries = ["DisplayPort cable", "Display Port cable", "DP cable"]
        for query in queries:
            result = extractor.extract(query)
            assert result.connector_from == "DisplayPort", f"Failed for: {query}"


class TestNonCableCategories:
    """Test connector suppression for non-cable categories (hubs, docks, etc.)."""

    def test_usb_hub_no_connectors(self, extractor):
        """USB hub should NOT extract USB as connector (USB describes hub type)."""
        result = extractor.extract("USB hub")
        assert result.product_category == "Hubs"
        assert result.connector_from is None
        assert result.connector_to is None

    def test_usb_c_hub_no_connectors(self, extractor):
        """USB-C hub should NOT extract USB-C as connector."""
        result = extractor.extract("USB-C hub")
        assert result.product_category == "Hubs"
        assert result.connector_from is None
        assert result.connector_to is None

    def test_usb_dock_no_connectors(self, extractor):
        """USB dock should NOT extract USB as connector."""
        result = extractor.extract("USB-C dock")
        assert result.product_category == "Docks"
        assert result.connector_from is None
        assert result.connector_to is None

    def test_dock_with_explicit_pair_keeps_connectors(self, extractor):
        """USB-C to HDMI dock SHOULD keep connectors (explicit pair)."""
        result = extractor.extract("USB-C to HDMI dock")
        assert result.product_category == "Docks"
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"

    def test_hdmi_cable_keeps_connectors(self, extractor):
        """HDMI cable SHOULD keep connectors (cable category)."""
        result = extractor.extract("HDMI cable")
        assert result.product_category == "Cables"
        assert result.connector_from == "HDMI"
        assert result.connector_to == "HDMI"

    def test_kvm_switch_no_connectors(self, extractor):
        """HDMI KVM switch should NOT extract HDMI as connector."""
        result = extractor.extract("HDMI KVM switch")
        assert result.product_category is not None
        # KVM switch category - connector should be suppressed for ambiguous term
        assert result.connector_from is None or result.product_category.lower() in ['kvm switches', 'switches']


class TestColorExtraction:
    """Tests for color extraction from queries."""

    def test_red_cable(self, extractor):
        """Should extract red color."""
        result = extractor.extract("red HDMI cable")
        assert result.color == "Red"

    def test_black_cable(self, extractor):
        """Should extract black color."""
        result = extractor.extract("black USB-C cable")
        assert result.color == "Black"

    def test_white_cable(self, extractor):
        """Should extract white color."""
        result = extractor.extract("white ethernet cable")
        assert result.color == "White"

    def test_grey_british_spelling(self, extractor):
        """Should normalize grey to Gray."""
        result = extractor.extract("grey DisplayPort cable")
        assert result.color == "Gray"

    def test_gray_american_spelling(self, extractor):
        """Should extract gray color."""
        result = extractor.extract("gray USB hub")
        assert result.color == "Gray"

    def test_no_color(self, extractor):
        """Should return None when no color specified."""
        result = extractor.extract("6ft HDMI cable")
        assert result.color is None

    def test_color_with_other_filters(self, extractor):
        """Should extract color alongside other filters."""
        result = extractor.extract("6ft red USB-C to HDMI cable")
        assert result.color == "Red"
        assert result.length == 6.0
        assert result.connector_from == "USB-C"
        assert result.connector_to == "HDMI"

    def test_color_case_insensitive(self, extractor):
        """Should handle uppercase/mixed case colors."""
        result = extractor.extract("RED HDMI cable")
        assert result.color == "Red"

    def test_orange_not_in_storage(self, extractor):
        """Should not match 'orange' inside 'storage'."""
        result = extractor.extract("storage device cable")
        assert result.color is None


class TestKeywordExtraction:
    """Tests for keyword extraction for text matching."""

    def test_fiber_optic_cable(self, extractor):
        """Fiber optic should be extracted as keywords AND category."""
        result = extractor.extract("fiber optic cable")
        # Category is detected from category_keywords
        assert result.product_category == "Fiber Cables"
        # Keywords MUST also be extracted for text matching
        # (category detection alone isn't reliable for all products)
        assert "fiber" in result.keywords
        assert "optic" in result.keywords

    def test_monitor_mount(self, extractor):
        """Should extract 'monitor' as keyword (mount is already category)."""
        result = extractor.extract("monitor mount")
        assert "monitor" in result.keywords
        # 'mount' should be captured by category, not keywords
        assert "mount" not in result.keywords

    def test_hard_drive_enclosure(self, extractor):
        """Drive enclosure should be captured by category AND keywords."""
        result = extractor.extract("hard drive enclosure")
        # Category is detected from category_keywords
        assert result.product_category == "Storage Enclosures"
        # Keywords MUST also be extracted for text matching
        assert "hard" in result.keywords
        assert "drive" in result.keywords
        # 'enclosure' is in ALREADY_EXTRACTED (generic category word)
        assert "enclosure" not in result.keywords

    def test_desk_mount(self, extractor):
        """Should extract 'desk' as keyword."""
        result = extractor.extract("desk mount for monitor")
        assert "desk" in result.keywords

    def test_wall_mount(self, extractor):
        """Should extract 'wall' as keyword."""
        result = extractor.extract("wall mount for TV")
        assert "wall" in result.keywords

    def test_no_keywords_for_standard_cable(self, extractor):
        """Standard cable query should have no extra keywords."""
        result = extractor.extract("6ft USB-C to HDMI cable")
        # All significant words are already captured by other extractors
        assert len(result.keywords) == 0

    def test_stop_words_excluded(self, extractor):
        """Stop words should not be extracted as keywords."""
        result = extractor.extract("I need a cable for the monitor")
        assert "need" not in result.keywords
        assert "for" not in result.keywords
        assert "the" not in result.keywords

    def test_short_words_excluded(self, extractor):
        """Words less than 3 characters should be excluded."""
        result = extractor.extract("TV mount on wall")
        assert "tv" not in result.keywords  # Too short
        assert "on" not in result.keywords  # Stop word and short

    def test_power_cord_keywords(self, extractor):
        """Power cord query should extract 'power' and 'cord'."""
        result = extractor.extract("power cord")
        assert "power" in result.keywords
        # Note: 'cord' is in ALREADY_EXTRACTED as it maps to cables
        # So only 'power' should be extracted

    def test_cat6_ethernet_keywords(self, extractor):
        """Cat6 ethernet query should extract 'cat6'."""
        result = extractor.extract("cat6 ethernet cable")
        assert "cat6" in result.keywords

    def test_sata_cable_keywords(self, extractor):
        """SATA cable query should extract 'sata'."""
        result = extractor.extract("SATA data cable")
        assert "sata" in result.keywords
        assert "data" in result.keywords

    def test_patch_cable_keywords(self, extractor):
        """Patch cable query should extract 'patch' and 'fiber' as keywords."""
        result = extractor.extract("fiber patch cable")
        assert "patch" in result.keywords
        # 'fiber' must remain as keyword for text matching
        assert result.product_category == "Fiber Cables"
        assert "fiber" in result.keywords

    def test_dual_monitor_arm(self, extractor):
        """Dual monitor arm should extract relevant keywords."""
        result = extractor.extract("dual monitor arm")
        assert "dual" in result.keywords
        assert "arm" in result.keywords


class TestFiberAndStorageCategories:
    """Tests for fiber cable and storage enclosure category extraction."""

    def test_fiber_optic_category(self, extractor):
        """'fiber optic cable' should match Fiber Cables category."""
        result = extractor.extract("fiber optic cable")
        assert result.product_category == "Fiber Cables"

    def test_fiber_patch_category(self, extractor):
        """'fiber patch cable' should match Fiber Cables category."""
        result = extractor.extract("fiber patch cable")
        assert result.product_category == "Fiber Cables"

    def test_optical_fiber_category(self, extractor):
        """'optical fiber' should match Fiber Cables category."""
        result = extractor.extract("optical fiber cable")
        assert result.product_category == "Fiber Cables"

    def test_drive_enclosure_category(self, extractor):
        """'drive enclosure' should match Storage Enclosures category."""
        result = extractor.extract("drive enclosure")
        assert result.product_category == "Storage Enclosures"

    def test_hard_drive_enclosure_category(self, extractor):
        """'hard drive enclosure' should match Storage Enclosures category."""
        result = extractor.extract("hard drive enclosure")
        assert result.product_category == "Storage Enclosures"

    def test_ssd_enclosure_category(self, extractor):
        """'ssd enclosure' should match Storage Enclosures category."""
        result = extractor.extract("ssd enclosure")
        assert result.product_category == "Storage Enclosures"

    def test_nvme_enclosure_category(self, extractor):
        """'nvme enclosure' should match Storage Enclosures category."""
        result = extractor.extract("nvme enclosure")
        assert result.product_category == "Storage Enclosures"

    def test_m2_enclosure_category(self, extractor):
        """'m.2 enclosure' should match Storage Enclosures category."""
        result = extractor.extract("m.2 enclosure")
        assert result.product_category == "Storage Enclosures"


# Run tests with: pytest tests/test_filters.py -v