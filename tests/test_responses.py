"""
Tests for response formatting module.

Run with: pytest tests/test_responses.py -v
"""

import pytest
from ui.responses import (
    ResponseFormatter,
    get_response_formatter
)
from core.context import Product


@pytest.fixture
def formatter():
    """Create ResponseFormatter instance."""
    return ResponseFormatter()


@pytest.fixture
def sample_products():
    """Create sample products for testing."""
    return [
        Product(
            product_number="CABLE001",
            content="6ft HDMI Cable with 4K support",
            metadata={
                'name': '6ft HDMI Cable',
                'length': 6.0,
                'length_unit': 'ft',
                'features': ['4K', 'HDCP'],
                'connectors': ['HDMI', 'HDMI']
            }
        ),
        Product(
            product_number="CABLE002",
            content="USB-C to HDMI Cable",
            metadata={
                'name': 'USB-C to HDMI Cable',
                'length': 3.0,
                'length_unit': 'ft',
                'features': ['4K'],
                'connectors': ['USB-C', 'HDMI']
            }
        ),
    ]


class TestResponseFormatter:
    """Test ResponseFormatter class."""
    
    def test_creation(self, formatter):
        """Test creating ResponseFormatter."""
        assert isinstance(formatter, ResponseFormatter)
        assert formatter.prompts is not None
    
    def test_format_greeting(self, formatter):
        """Test formatting greeting."""
        greeting = formatter.format_greeting()
        
        assert isinstance(greeting, str)
        assert len(greeting) > 10


class TestProductFormatting:
    """Test product response formatting."""
    
    def test_format_product_response_with_products(self, formatter, sample_products):
        """Test formatting response with products."""
        response = formatter.format_product_response(
            products=sample_products,
            query="HDMI cable"
        )
        
        assert isinstance(response, str)
        assert "HDMI cable" in response
        assert "CABLE001" in response
        assert "CABLE002" in response
    
    def test_format_product_response_no_products(self, formatter):
        """Test formatting response with no products."""
        response = formatter.format_product_response(
            products=[],
            query="test query"
        )
        
        assert isinstance(response, str)
        assert "test query" in response
    
    def test_format_product_response_with_context_note(self, formatter, sample_products):
        """Test formatting response with context note."""
        response = formatter.format_product_response(
            products=sample_products,
            query="test",
            context_note="💡 Tip: This is a tip"
        )
        
        assert "💡 Tip: This is a tip" in response
    
    def test_format_product_response_with_tier(self, formatter, sample_products):
        """Test formatting response with search tier."""
        response = formatter.format_product_response(
            products=sample_products,
            query="test",
            tier="tier2"
        )
        
        assert "tier2" in response.lower()
    
    def test_format_single_product(self, formatter, sample_products):
        """Test formatting single product."""
        result = formatter._format_single_product(sample_products[0], 1)

        assert "6ft HDMI Cable" in result
        assert "CABLE001" in result
        assert "6.0ft" in result
        assert "4K" in result

    def test_format_pcie_network_card(self):
        """Test formatting PCIe network card shows card specs, not cable format."""
        from llm.response_builder import ResponseBuilder
        builder = ResponseBuilder()

        # Create a PCIe network card product
        pcie_card = Product(
            product_number="ST1000SPEX2",
            content="4-Port Gigabit Ethernet Network Card",
            metadata={
                'category': 'computer_card',
                'sub_category': 'Desktop and Server Network Cards',
                'BUSTYPE': 'PCI Express x1',
                'CARDPROFILE': 'Low Profile',
                'NUMBERPORTS': 4,
                'INTERFACEA': '1 x PCI Express x1',
                'INTERFACEB': '4 x RJ-45 (Gigabit Ethernet)',
                'features': ['Gigabit Ethernet']
            }
        )

        result = builder._format_pcie_card_line(pcie_card, 1)

        # Should show card-specific format
        assert "ST1000SPEX2" in result
        assert "PCIe x1" in result or "PCI Express x1" in result  # Accepts both formats
        assert "Network Card" in result
        assert "Low Profile" in result
        assert "Gigabit" in result  # Should show network speed
        # Should NOT show cable-style formatting
        assert "ft" not in result.lower()  # No length

    def test_is_pcie_card_detection(self):
        """Test PCIe card detection works correctly."""
        from llm.response_builder import ResponseBuilder
        builder = ResponseBuilder()

        # Computer card category
        card1 = Product("CARD1", "", metadata={'category': 'computer_card'})
        assert builder._is_pcie_card(card1) is True

        # Has PCI in BUSTYPE
        card2 = Product("CARD2", "", metadata={'BUSTYPE': 'PCI Express x4'})
        assert builder._is_pcie_card(card2) is True

        # Network card sub_category
        card3 = Product("CARD3", "", metadata={'sub_category': 'Desktop and Server Network Cards'})
        assert builder._is_pcie_card(card3) is True

        # Regular cable - should NOT be detected as card
        cable = Product("CABLE1", "", metadata={'category': 'cable', 'length_ft': 6})
        assert builder._is_pcie_card(cable) is False

    def test_format_multiport_adapter(self):
        """Test formatting multiport adapter shows port config, not cable format."""
        from llm.response_builder import ResponseBuilder
        builder = ResponseBuilder()

        # Create a USB-C multiport adapter product with EXTERNALPORTS field
        adapter = Product(
            product_number="DKT30CHPD3",
            content="USB-C Multiport Adapter with HDMI, USB 3.0, and Gigabit Ethernet with 100W Power Delivery",
            metadata={
                'category': 'multiport_adapter',
                'sub_category': 'USB-C Multiport Adapters',
                'EXTERNALPORTS': '1 x HDMI, 1 x RJ-45, 2 x USB 3.2 Type-A, 1 x MicroSD, 1 x SD / MMC Slot',
                'POWERDELIVERY': 'Yes',
                'DOCK4KSUPPORT': 'Yes',
                'features': ['4K', 'Power Delivery']
            }
        )

        result = builder._format_multiport_adapter_line(adapter, 1)

        # Should show adapter-specific format
        assert "DKT30CHPD3" in result
        assert "USB-C" in result  # Input type
        assert "Multiport Adapter" in result
        assert "HDMI" in result  # Video output from EXTERNALPORTS
        assert "USB-A" in result  # USB ports from EXTERNALPORTS
        assert "GbE" in result  # Ethernet from EXTERNALPORTS
        # Should show Power Delivery
        assert "PD" in result
        # Should show 4K
        assert "4K" in result
        # Should NOT show cable-style formatting
        assert "ft" not in result.lower()  # No length

    def test_is_multiport_adapter_detection(self):
        """Test multiport adapter detection works correctly."""
        from llm.response_builder import ResponseBuilder
        builder = ResponseBuilder()

        # Multiport adapter category
        adapter1 = Product("ADAPT1", "", metadata={'category': 'multiport_adapter'})
        assert builder._is_multiport_adapter(adapter1) is True

        # Has multiport in sub_category
        adapter2 = Product("ADAPT2", "", metadata={'sub_category': 'USB-C Multiport Adapters'})
        assert builder._is_multiport_adapter(adapter2) is True

        # Has MULTIPORT in SKU
        adapter3 = Product("102B-USBC-MULTIPORT", "", metadata={})
        assert builder._is_multiport_adapter(adapter3) is True

        # Regular cable - should NOT be detected as adapter
        cable = Product("CABLE1", "", metadata={'category': 'cable', 'length_ft': 6})
        assert builder._is_multiport_adapter(cable) is False


class TestConversationFormatting:
    """Test conversation response formatting."""
    
    def test_format_greeting(self, formatter):
        """Test formatting greeting."""
        greeting = formatter.format_greeting()
        
        assert isinstance(greeting, str)
        assert len(greeting) > 0
    
    def test_format_farewell(self, formatter):
        """Test formatting farewell."""
        farewell = formatter.format_farewell()
        
        assert isinstance(farewell, str)
        assert len(farewell) > 0
    
    def test_format_ambiguous_query(self, formatter):
        """Test formatting ambiguous query response."""
        response = formatter.format_ambiguous_query()
        
        assert isinstance(response, str)
        assert len(response) > 20


class TestBlockedAndErrors:
    """Test blocked request and error formatting."""
    
    def test_format_blocked_request_basic(self, formatter):
        """Test formatting blocked request without alternatives."""
        response = formatter.format_blocked_request("Not supported")
        
        assert "Not supported" in response
    
    def test_format_blocked_request_with_alternatives(self, formatter):
        """Test formatting blocked request with alternatives."""
        response = formatter.format_blocked_request(
            reason="Not supported",
            alternatives=["Option 1", "Option 2"]
        )
        
        assert "Not supported" in response
        assert "Option 1" in response
        assert "Option 2" in response
    
    def test_format_no_results(self, formatter):
        """Test formatting no results response."""
        response = formatter.format_no_results("test query")
        
        assert "test query" in response
    
    def test_format_no_results_with_suggestions(self, formatter):
        """Test formatting no results with suggestions."""
        response = formatter.format_no_results(
            query="test query",
            suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        assert "test query" in response
        assert "Suggestion 1" in response
        assert "Suggestion 2" in response
    
    def test_format_error(self, formatter):
        """Test formatting error response."""
        error = formatter.format_error("search_failed")
        
        assert isinstance(error, str)
        assert len(error) > 0


class TestContextNotes:
    """Test context note formatting."""
    
    def test_format_with_context_note(self, formatter):
        """Test adding context note to response."""
        response = formatter.format_with_context_note(
            main_response="Main content",
            context_type="4k"
        )
        
        assert "Main content" in response
        assert "💡" in response or "Tip" in response
    
    def test_format_with_context_note_details(self, formatter):
        """Test adding context note with details."""
        response = formatter.format_with_context_note(
            main_response="Main content",
            context_type="4k",
            details="Extra info"
        )
        
        assert "Main content" in response
        assert "Extra info" in response


class TestInfoFormatting:
    """Test connector and feature info formatting."""
    
    def test_format_connector_info(self, formatter):
        """Test formatting connector information."""
        info = formatter.format_connector_info("USB-C")
        
        assert isinstance(info, str)
        assert "USB-C" in info
    
    def test_format_feature_info(self, formatter):
        """Test formatting feature information."""
        info = formatter.format_feature_info("4K")
        
        assert isinstance(info, str)
        assert "4K" in info


class TestTextUtilities:
    """Test text utility methods."""
    
    def test_format_multi_line_no_indent(self, formatter):
        """Test multi-line formatting without indent."""
        text = "Line 1\nLine 2"
        result = formatter.format_multi_line(text, indent=0)
        
        assert result == text
    
    def test_format_multi_line_with_indent(self, formatter):
        """Test multi-line formatting with indent."""
        text = "Line 1\nLine 2"
        result = formatter.format_multi_line(text, indent=2)
        
        assert result.startswith("  ")
        assert "\n  " in result
    
    def test_truncate_text_short(self, formatter):
        """Test truncating short text."""
        text = "Short text"
        result = formatter.truncate_text(text, max_length=50)
        
        assert result == text
    
    def test_truncate_text_long(self, formatter):
        """Test truncating long text."""
        text = "This is a very long text that needs to be truncated"
        result = formatter.truncate_text(text, max_length=20)
        
        assert len(result) == 20
        assert result.endswith("...")


class TestSingletonAccess:
    """Test singleton accessor."""
    
    def test_get_response_formatter(self):
        """Test getting formatter singleton."""
        formatter = get_response_formatter()
        
        assert isinstance(formatter, ResponseFormatter)
    
    def test_singleton_same_instance(self):
        """Test that singleton returns same instance."""
        formatter1 = get_response_formatter()
        formatter2 = get_response_formatter()
        
        assert formatter1 is formatter2


# Run tests with: pytest tests/test_responses.py -v