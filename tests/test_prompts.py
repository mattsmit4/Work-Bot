"""
Tests for system prompts module.

Run with: pytest tests/test_prompts.py -v
"""

import pytest
from llm.prompts import (
    SystemPrompts,
    ResponseTemplates,
    get_system_prompts,
    get_response_templates
)


@pytest.fixture
def prompts():
    """Create SystemPrompts instance."""
    return SystemPrompts()


@pytest.fixture
def templates():
    """Create ResponseTemplates instance."""
    return ResponseTemplates()


class TestSystemPrompts:
    """Test SystemPrompts class."""
    
    def test_creation(self, prompts):
        """Test creating SystemPrompts."""
        assert isinstance(prompts, SystemPrompts)
    
    def test_query_parser_prompt(self, prompts):
        """Test getting query parser prompt."""
        prompt = prompts.get_query_parser_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "JSON" in prompt
        assert "product_type" in prompt
    
    def test_conceptual_question_prompt(self, prompts):
        """Test getting conceptual question prompt."""
        prompt = prompts.get_conceptual_question_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "technical expert" in prompt.lower() or "expert" in prompt.lower()


class TestBlockedRequests:
    """Test blocked request formatting."""
    
    def test_format_blocked_request_basic(self, prompts):
        """Test formatting blocked request without alternatives."""
        response = prompts.format_blocked_request("Not supported")
        
        assert "Not supported" in response
        assert isinstance(response, str)
    
    def test_format_blocked_request_with_alternatives(self, prompts):
        """Test formatting blocked request with alternatives."""
        alternatives = ["Option 1", "Option 2"]
        response = prompts.format_blocked_request(
            "Not supported",
            alternatives=alternatives
        )
        
        assert "Not supported" in response
        assert "Option 1" in response
        assert "Option 2" in response
        assert "Alternatives" in response or "alternatives" in response.lower()


class TestGreetingsAndFarewells:
    """Test greeting and farewell responses."""
    
    def test_format_greeting(self, prompts):
        """Test formatting greeting response."""
        response = prompts.format_greeting_response()

        assert isinstance(response, str)
        assert len(response) > 10
        # Should be a question or invitation to start
        assert "?" in response
    
    def test_format_farewell(self, prompts):
        """Test formatting farewell response."""
        response = prompts.format_farewell_response()
        
        assert isinstance(response, str)
        assert len(response) > 10


class TestNoResults:
    """Test no results responses."""
    
    def test_format_no_results_basic(self, prompts):
        """Test formatting no results without suggestions."""
        response = prompts.format_no_results_response("test query")
        
        assert "test query" in response
        assert isinstance(response, str)
    
    def test_format_no_results_with_suggestions(self, prompts):
        """Test formatting no results with suggestions."""
        suggestions = ["Try this", "Try that"]
        response = prompts.format_no_results_response(
            "test query",
            suggestions=suggestions
        )
        
        assert "test query" in response
        assert "Try this" in response
        assert "Try that" in response


class TestAmbiguousQuery:
    """Test ambiguous query responses."""
    
    def test_format_ambiguous_query(self, prompts):
        """Test formatting ambiguous query response."""
        response = prompts.format_ambiguous_query_response()
        
        assert isinstance(response, str)
        assert len(response) > 30
        assert "information" in response.lower() or "help" in response.lower()


class TestProductSummary:
    """Test product summary formatting."""
    
    def test_format_product_summary_zero(self, prompts):
        """Test formatting summary with zero products."""
        summary = prompts.format_product_summary(0, "test")
        
        assert "No products" in summary or "0" in summary
        assert "test" in summary
    
    def test_format_product_summary_one(self, prompts):
        """Test formatting summary with one product."""
        summary = prompts.format_product_summary(1, "test")
        
        assert "1" in summary
        assert "test" in summary
    
    def test_format_product_summary_many(self, prompts):
        """Test formatting summary with multiple products."""
        summary = prompts.format_product_summary(5, "test")
        
        assert "5" in summary
        assert "test" in summary


class TestContextNotes:
    """Test context note formatting."""
    
    def test_format_context_note_4k(self, prompts):
        """Test formatting 4K context note."""
        note = prompts.format_context_note("4k")
        
        assert isinstance(note, str)
        assert "4K" in note or "4k" in note
    
    def test_format_context_note_long_cable(self, prompts):
        """Test formatting long cable context note."""
        note = prompts.format_context_note("long_cable")
        
        assert isinstance(note, str)
        assert "15ft" in note or "long" in note.lower()
    
    def test_format_context_note_with_details(self, prompts):
        """Test formatting context note with details."""
        note = prompts.format_context_note("4k", "Additional info")
        
        assert "Additional info" in note
    
    def test_format_context_note_unknown_type(self, prompts):
        """Test formatting context note with unknown type."""
        note = prompts.format_context_note("unknown_type")
        
        assert isinstance(note, str)


class TestErrorResponses:
    """Test error response formatting."""
    
    def test_format_error_search_failed(self, prompts):
        """Test formatting search failed error."""
        response = prompts.format_error_response("search_failed")
        
        assert isinstance(response, str)
        assert len(response) > 10
    
    def test_format_error_invalid_input(self, prompts):
        """Test formatting invalid input error."""
        response = prompts.format_error_response("invalid_input")
        
        assert isinstance(response, str)
        assert len(response) > 10
    
    def test_format_error_unknown_type(self, prompts):
        """Test formatting unknown error type."""
        response = prompts.format_error_response("unknown_error")
        
        assert isinstance(response, str)
        assert "error" in response.lower()


class TestResponseTemplates:
    """Test ResponseTemplates class."""
    
    def test_format_connector_explanation_usb_c(self, templates):
        """Test USB-C connector explanation."""
        explanation = templates.format_connector_explanation("USB-C")
        
        assert isinstance(explanation, str)
        assert "USB-C" in explanation
        assert len(explanation) > 20
    
    def test_format_connector_explanation_hdmi(self, templates):
        """Test HDMI connector explanation."""
        explanation = templates.format_connector_explanation("HDMI")
        
        assert isinstance(explanation, str)
        assert "HDMI" in explanation
    
    def test_format_connector_explanation_unknown(self, templates):
        """Test unknown connector explanation."""
        explanation = templates.format_connector_explanation("Unknown")
        
        assert isinstance(explanation, str)
        assert "Unknown" in explanation
    
    def test_format_feature_explanation_4k(self, templates):
        """Test 4K feature explanation."""
        explanation = templates.format_feature_explanation("4K")
        
        assert isinstance(explanation, str)
        assert "4K" in explanation
        assert len(explanation) > 20
    
    def test_format_feature_explanation_hdr(self, templates):
        """Test HDR feature explanation."""
        explanation = templates.format_feature_explanation("HDR")
        
        assert isinstance(explanation, str)
        assert "HDR" in explanation
    
    def test_format_feature_explanation_unknown(self, templates):
        """Test unknown feature explanation."""
        explanation = templates.format_feature_explanation("Unknown")
        
        assert isinstance(explanation, str)
        assert "Unknown" in explanation


class TestSingletonAccess:
    """Test singleton accessor functions."""
    
    def test_get_system_prompts(self):
        """Test getting system prompts singleton."""
        prompts = get_system_prompts()
        
        assert isinstance(prompts, SystemPrompts)
    
    def test_get_response_templates(self):
        """Test getting response templates singleton."""
        templates = get_response_templates()
        
        assert isinstance(templates, ResponseTemplates)
    
    def test_singleton_same_instance(self):
        """Test that singleton returns same instance."""
        prompts1 = get_system_prompts()
        prompts2 = get_system_prompts()
        
        assert prompts1 is prompts2


# Run tests with: pytest tests/test_prompts.py -v