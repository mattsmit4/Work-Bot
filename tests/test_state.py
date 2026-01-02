"""
Tests for session state management module.

Run with: pytest tests/test_state.py -v
"""

import pytest
from datetime import datetime
from ui.state import SessionState, Message, get_session_state
from core.context import Product, IntentType


@pytest.fixture
def state():
    """Create a fresh SessionState instance."""
    return SessionState()


@pytest.fixture
def sample_products():
    """Create sample products for testing."""
    return [
        Product(
            product_number="CABLE001",
            content="6ft HDMI Cable",
            metadata={'name': '6ft HDMI Cable'}
        ),
        Product(
            product_number="CABLE002",
            content="USB-C Cable",
            metadata={'name': 'USB-C Cable'}
        ),
    ]


class TestSessionState:
    """Test SessionState class."""
    
    def test_creation(self, state):
        """Test creating SessionState."""
        assert isinstance(state, SessionState)
        assert state.session_id is not None
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)
    
    def test_custom_session_id(self):
        """Test creating SessionState with custom ID."""
        state = SessionState(session_id="test_123")
        assert state.session_id == "test_123"
    
    def test_session_id_generation(self, state):
        """Test automatic session ID generation."""
        assert state.session_id.startswith("session_")


class TestMessages:
    """Test message management."""
    
    def test_add_message(self, state):
        """Test adding a message."""
        message = state.add_message("user", "Hello")
        
        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.content == "Hello"
        assert isinstance(message.timestamp, datetime)
    
    def test_add_message_with_metadata(self, state):
        """Test adding message with metadata."""
        metadata = {'intent': 'greeting'}
        message = state.add_message("user", "Hi", metadata=metadata)
        
        assert message.metadata == metadata
    
    def test_get_conversation_history(self, state):
        """Test getting conversation history."""
        state.add_message("user", "Message 1")
        state.add_message("assistant", "Message 2")
        state.add_message("user", "Message 3")
        
        history = state.get_conversation_history()
        
        assert len(history) == 3
        assert history[0].content == "Message 1"
        assert history[1].content == "Message 2"
        assert history[2].content == "Message 3"
    
    def test_get_conversation_history_with_limit(self, state):
        """Test getting conversation history with limit."""
        for i in range(10):
            state.add_message("user", f"Message {i}")
        
        history = state.get_conversation_history(limit=3)
        
        assert len(history) == 3
        assert history[0].content == "Message 7"
        assert history[2].content == "Message 9"
    
    def test_get_conversation_history_by_role(self, state):
        """Test filtering conversation history by role."""
        state.add_message("user", "User 1")
        state.add_message("assistant", "Assistant 1")
        state.add_message("user", "User 2")
        state.add_message("assistant", "Assistant 2")
        
        user_msgs = state.get_conversation_history(role='user')
        assistant_msgs = state.get_conversation_history(role='assistant')
        
        assert len(user_msgs) == 2
        assert len(assistant_msgs) == 2
        assert all(m.role == 'user' for m in user_msgs)
        assert all(m.role == 'assistant' for m in assistant_msgs)
    
    def test_get_last_message(self, state):
        """Test getting last message."""
        state.add_message("user", "First")
        state.add_message("assistant", "Second")
        
        last = state.get_last_message()
        
        assert last.content == "Second"
        assert last.role == "assistant"
    
    def test_get_last_message_by_role(self, state):
        """Test getting last message by role."""
        state.add_message("user", "User 1")
        state.add_message("assistant", "Assistant 1")
        state.add_message("user", "User 2")
        
        last_user = state.get_last_message(role='user')
        last_assistant = state.get_last_message(role='assistant')
        
        assert last_user.content == "User 2"
        assert last_assistant.content == "Assistant 1"
    
    def test_get_last_message_empty(self, state):
        """Test getting last message from empty history."""
        last = state.get_last_message()
        assert last is None
    
    def test_clear_messages(self, state):
        """Test clearing messages."""
        state.add_message("user", "Test")
        state.clear_messages()
        
        assert len(state.get_conversation_history()) == 0


class TestProductContext:
    """Test product context management."""
    
    def test_set_product_context(self, state, sample_products):
        """Test setting product context."""
        state.set_product_context(sample_products)
        
        context = state.get_product_context()
        assert len(context.current_products) == 2
    
    def test_set_product_context_with_intent(self, state, sample_products):
        """Test setting product context with intent."""
        state.set_product_context(sample_products, IntentType.NEW_SEARCH)
        
        context = state.get_product_context()
        assert len(context.current_products) == 2
    
    def test_get_product_context(self, state):
        """Test getting product context."""
        context = state.get_product_context()
        
        assert context is not None
        assert context.current_products is None
        assert context.last_product is None
    
    def test_clear_product_context(self, state, sample_products):
        """Test clearing product context."""
        state.set_product_context(sample_products)
        state.clear_product_context()
        
        context = state.get_product_context()
        assert context.current_products is None
        assert context.last_product is None


class TestPreferences:
    """Test user preference management."""
    
    def test_set_preference(self, state):
        """Test setting a preference."""
        state.set_preference("display_count", 5)
        
        value = state.get_preference("display_count")
        assert value == 5
    
    def test_get_preference_default(self, state):
        """Test getting preference with default."""
        value = state.get_preference("nonexistent", default=10)
        assert value == 10
    
    def test_get_preference_not_found(self, state):
        """Test getting non-existent preference."""
        value = state.get_preference("nonexistent")
        assert value is None
    
    def test_clear_preferences(self, state):
        """Test clearing preferences."""
        state.set_preference("key1", "value1")
        state.set_preference("key2", "value2")
        state.clear_preferences()
        
        assert state.get_preference("key1") is None
        assert state.get_preference("key2") is None


class TestMetadata:
    """Test session metadata management."""
    
    def test_set_metadata(self, state):
        """Test setting metadata."""
        state.set_metadata("user_id", "user123")
        
        value = state.get_metadata("user_id")
        assert value == "user123"
    
    def test_get_metadata_default(self, state):
        """Test getting metadata with default."""
        value = state.get_metadata("nonexistent", default="default")
        assert value == "default"
    
    def test_get_metadata_not_found(self, state):
        """Test getting non-existent metadata."""
        value = state.get_metadata("nonexistent")
        assert value is None


class TestSessionInfo:
    """Test session information methods."""
    
    def test_get_message_count(self, state):
        """Test getting message count."""
        assert state.get_message_count() == 0
        
        state.add_message("user", "Test 1")
        state.add_message("assistant", "Test 2")
        
        assert state.get_message_count() == 2
    
    def test_get_session_duration(self, state):
        """Test getting session duration."""
        duration = state.get_session_duration()
        
        assert isinstance(duration, float)
        assert duration >= 0


class TestReset:
    """Test session reset functionality."""
    
    def test_reset(self, state, sample_products):
        """Test resetting session."""
        # Add data
        state.add_message("user", "Test")
        state.set_product_context(sample_products)
        state.set_preference("key", "value")
        state.set_metadata("meta", "data")
        
        # Reset
        old_session_id = state.session_id
        old_created_at = state.created_at
        state.reset()
        
        # Verify cleared
        assert state.get_message_count() == 0
        context = state.get_product_context()
        assert context.current_products is None
        assert context.last_product is None
        assert state.get_preference("key") is None
        assert state.get_metadata("meta") is None
        
        # Verify kept
        assert state.session_id == old_session_id
        assert state.created_at == old_created_at


class TestExport:
    """Test exporting session state."""
    
    def test_to_dict(self, state, sample_products):
        """Test exporting to dictionary."""
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi")
        state.set_product_context(sample_products)
        state.set_preference("display_count", 5)
        state.set_metadata("user_id", "user123")
        
        state_dict = state.to_dict()
        
        assert state_dict['session_id'] == state.session_id
        assert state_dict['message_count'] == 2
        assert len(state_dict['messages']) == 2
        assert state_dict['product_count'] == 2
        assert state_dict['preferences']['display_count'] == 5
        assert state_dict['metadata']['user_id'] == "user123"


class TestSingletonAccess:
    """Test singleton accessor."""
    
    def test_get_session_state(self):
        """Test getting session state singleton."""
        state = get_session_state(reset=True)
        
        assert isinstance(state, SessionState)
    
    def test_singleton_same_instance(self):
        """Test that singleton returns same instance."""
        state1 = get_session_state(reset=True)
        state2 = get_session_state()
        
        assert state1 is state2
    
    def test_singleton_reset(self):
        """Test resetting singleton."""
        state1 = get_session_state(reset=True)
        state1.add_message("user", "Test")
        
        state2 = get_session_state(reset=True)
        
        assert state2.get_message_count() == 0


class TestIntegration:
    """Test integration scenarios."""
    
    def test_full_conversation_flow(self, state, sample_products):
        """Test full conversation workflow."""
        # User greeting
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi! How can I help?")
        
        # User search
        state.add_message("user", "I need HDMI cables")
        state.set_product_context(sample_products, IntentType.NEW_SEARCH)
        state.add_message("assistant", "Here are 2 products...")
        
        # User follow-up
        state.add_message("user", "Tell me about the first one")
        
        # Verify state
        assert state.get_message_count() == 5
        assert len(state.get_product_context().current_products) == 2
        
        last_user = state.get_last_message(role='user')
        assert "first one" in last_user.content


# Run tests with: pytest tests/test_state.py -v