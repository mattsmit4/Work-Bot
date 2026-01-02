"""
Tests for simplified intent classification.

Tests the 5 core intent types:
- GREETING
- FAREWELL
- NEW_SEARCH
- FOLLOWUP
- AMBIGUOUS
"""

import pytest
from core.intent import IntentClassifier
from core.context import ConversationContext, IntentType, Product


@pytest.fixture
def classifier():
    return IntentClassifier()


@pytest.fixture
def context():
    return ConversationContext()


@pytest.fixture
def context_with_products():
    """Context with products to test followup detection."""
    ctx = ConversationContext()
    products = [
        Product(
            product_number="CDP2HDUACP2",
            content="USB-C to HDMI adapter",
            metadata={"category": "adapter", "connectors": ["USB-C", "HDMI"]}
        )
    ]
    ctx.set_multi_products(products)
    return ctx


# === GREETING TESTS ===

class TestGreeting:
    """Test greeting intent detection."""

    def test_hello(self, classifier, context):
        intent = classifier.classify("Hello", context)
        assert intent.type == IntentType.GREETING

    def test_hi(self, classifier, context):
        intent = classifier.classify("Hi", context)
        assert intent.type == IntentType.GREETING

    def test_hey(self, classifier, context):
        intent = classifier.classify("Hey", context)
        assert intent.type == IntentType.GREETING

    def test_hello_there(self, classifier, context):
        intent = classifier.classify("Hello there", context)
        assert intent.type == IntentType.GREETING

    def test_long_greeting_not_detected(self, classifier, context):
        # Greetings are only detected for short messages (≤4 words)
        intent = classifier.classify("Hello I need a cable for my monitor", context)
        assert intent.type != IntentType.GREETING


# === FAREWELL TESTS ===

class TestFarewell:
    """Test farewell intent detection."""

    def test_goodbye(self, classifier, context):
        intent = classifier.classify("Goodbye", context)
        assert intent.type == IntentType.FAREWELL

    def test_bye(self, classifier, context):
        intent = classifier.classify("Bye", context)
        assert intent.type == IntentType.FAREWELL

    def test_thanks(self, classifier, context):
        intent = classifier.classify("Thanks, bye!", context)
        assert intent.type == IntentType.FAREWELL


# === NEW SEARCH TESTS ===

class TestNewSearch:
    """Test new product search intent detection."""

    def test_show_me_cables(self, classifier, context):
        intent = classifier.classify("Show me HDMI cables", context)
        assert intent.type == IntentType.NEW_SEARCH

    def test_connector_to_connector(self, classifier, context):
        intent = classifier.classify("USB-C to HDMI adapter", context)
        assert intent.type == IntentType.NEW_SEARCH

    def test_product_type_with_connector(self, classifier, context):
        intent = classifier.classify("I need an HDMI cable", context)
        assert intent.type == IntentType.NEW_SEARCH

    def test_length_with_domain(self, classifier, context):
        intent = classifier.classify("6ft DisplayPort cable", context)
        assert intent.type == IntentType.NEW_SEARCH

    def test_looking_for(self, classifier, context):
        intent = classifier.classify("I'm looking for a USB-C dock", context)
        assert intent.type == IntentType.NEW_SEARCH

    def test_dock_search(self, classifier, context):
        intent = classifier.classify("I need a docking station", context)
        assert intent.type == IntentType.NEW_SEARCH

    def test_kvm_search(self, classifier, context):
        intent = classifier.classify("Show me KVM switches", context)
        assert intent.type == IntentType.NEW_SEARCH


# === FOLLOWUP TESTS ===

class TestFollowup:
    """Test followup intent detection (when products are in context)."""

    def test_does_it_support(self, classifier, context_with_products):
        intent = classifier.classify("Does it support 4K?", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_which_one(self, classifier, context_with_products):
        intent = classifier.classify("Which one is best?", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_tell_me_more(self, classifier, context_with_products):
        intent = classifier.classify("Tell me more about it", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_difference(self, classifier, context_with_products):
        intent = classifier.classify("What's the difference?", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_shorter_please(self, classifier, context_with_products):
        intent = classifier.classify("shorter please", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_short_query_with_context(self, classifier, context_with_products):
        # Short queries with context are treated as followups
        intent = classifier.classify("4K support?", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_product_reference(self, classifier, context_with_products):
        intent = classifier.classify("Tell me about product 1", context_with_products)
        assert intent.type == IntentType.FOLLOWUP

    def test_new_search_overrides_context(self, classifier, context_with_products):
        # Even with context, explicit new search should be NEW_SEARCH
        intent = classifier.classify("Show me DisplayPort cables instead", context_with_products)
        assert intent.type == IntentType.NEW_SEARCH


# === AMBIGUOUS TESTS ===

class TestAmbiguous:
    """Test ambiguous intent detection."""

    def test_vague_query_no_context(self, classifier, context):
        # Vague query without product context or domain tokens
        intent = classifier.classify("I'm not sure what I need", context)
        assert intent.type == IntentType.AMBIGUOUS

    def test_random_text(self, classifier, context):
        intent = classifier.classify("banana apple orange", context)
        assert intent.type == IntentType.AMBIGUOUS
