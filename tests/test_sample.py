"""
Sample test file for ST-Bot.

Demonstrates how to write tests for the clean architecture.
Run with: pytest tests/test_sample.py -v
"""

import pytest
from core.context import (
    IntentType,
    Intent,
    Product,
    ConversationContext,
    FilterConfig,
)
from config.synonyms import expand_synonyms
from config.patterns import extract_lengths, has_pattern, GREETING_PATTERNS


class TestSynonyms:
    """Test synonym expansion."""
    
    def test_expand_dp_abbreviation(self):
        """Test that 'dp' expands to 'displayport'."""
        result = expand_synonyms("I need a 6ft DP cable")
        assert "displayport" in result
        assert "dp" not in result
    
    def test_expand_usb_c_variations(self):
        """Test USB-C variations normalize correctly."""
        inputs = [
            "usb c cable",
            "usbc cable",
            "type c cable",
        ]
        
        for text in inputs:
            result = expand_synonyms(text)
            assert "usb-c" in result
    
    def test_preserves_4k(self):
        """Test that '4k' is expanded to resolution."""
        result = expand_synonyms("4k monitor cable")
        assert "3840x2160" in result or "4k" in result


class TestPatterns:
    """Test regex patterns."""
    
    def test_extract_lengths(self):
        """Test length extraction from text."""
        text = "I need a cable between 3ft and 6ft"
        lengths = extract_lengths(text)
        
        assert len(lengths) == 2
        assert (3.0, 'ft') in lengths
        assert (6.0, 'ft') in lengths
    
    def test_greeting_detection(self):
        """Test greeting pattern detection."""
        assert has_pattern("Hello there!", GREETING_PATTERNS)
        assert has_pattern("Hi", GREETING_PATTERNS)
        assert not has_pattern("HDMI cable", GREETING_PATTERNS)


class TestDataModels:
    """Test data models."""
    
    def test_intent_creation(self):
        """Test creating an Intent object."""
        intent = Intent(
            type=IntentType.GREETING,
            confidence=1.0,
            reasoning="User said hello"
        )
        
        assert intent.type == IntentType.GREETING
        assert intent.confidence == 1.0
        assert intent.sku is None
    
    def test_intent_with_sku(self):
        """Test Intent with product SKU."""
        intent = Intent(
            type=IntentType.NEW_SEARCH,
            confidence=1.0,
            reasoning="User mentioned product number",
            sku="CDP2DPMM6B"
        )

        assert intent.sku == "CDP2DPMM6B"
    
    def test_conversation_context_multi_product(self):
        """Test ConversationContext with multiple products."""
        products = [
            Product("SKU1", "content1", {}, 0.9),
            Product("SKU2", "content2", {}, 0.8),
        ]
        
        context = ConversationContext()
        context.set_multi_products(products)
        
        assert context.has_multi_product_context()
        assert not context.has_single_product_context()
        assert len(context.current_products) == 2
    
    def test_conversation_context_single_product(self):
        """Test ConversationContext with single product."""
        product = Product("SKU1", "content", {}, 1.0)
        
        context = ConversationContext()
        context.set_single_product(product)
        
        assert context.has_single_product_context()
        assert not context.has_multi_product_context()
        assert context.last_product.product_number == "SKU1"
    
    def test_context_clear_products(self):
        """Test clearing product context."""
        context = ConversationContext()
        context.set_single_product(Product("SKU1", "content", {}, 1.0))
        
        context.clear_products()
        
        assert not context.has_single_product_context()
        assert not context.has_multi_product_context()


class TestProduct:
    """Test Product model."""
    
    def test_product_creation(self):
        """Test creating a Product."""
        product = Product(
            product_number="CDP2DPMM6B",
            content="USB-C to DisplayPort cable",
            metadata={"category": "cables", "length": 1800},
            score=0.95
        )
        
        assert product.product_number == "CDP2DPMM6B"
        assert product.score == 0.95
    
    def test_product_get_metadata(self):
        """Test getting metadata safely."""
        product = Product(
            "SKU1",
            "content",
            {"category": "cables", "color": "black"},
            1.0
        )
        
        assert product.get("category") == "cables"
        assert product.get("missing_key", "default") == "default"
        assert product.get("color") == "black"


# Fixtures for reusable test data
@pytest.fixture
def sample_context():
    """Create a sample conversation context."""
    return ConversationContext(
        query_count=5,
        session_id="test-session-123"
    )


@pytest.fixture
def sample_products():
    """Create sample products."""
    return [
        Product(
            "CDP2DPMM6B",
            "USB-C to DisplayPort Cable - 6ft",
            {"category": "cables", "subcategory": "display cables"},
            0.95
        ),
        Product(
            "CDP2DPMM1MB",
            "USB-C to DisplayPort Cable - 3ft",
            {"category": "cables", "subcategory": "display cables"},
            0.90
        ),
    ]


class TestWithFixtures:
    """Test using pytest fixtures."""
    
    def test_context_fixture(self, sample_context):
        """Test using sample context fixture."""
        assert sample_context.query_count == 5
        assert sample_context.session_id == "test-session-123"
    
    def test_products_fixture(self, sample_products):
        """Test using sample products fixture."""
        assert len(sample_products) == 2
        assert sample_products[0].product_number == "CDP2DPMM6B"
        assert sample_products[1].product_number == "CDP2DPMM1MB"
