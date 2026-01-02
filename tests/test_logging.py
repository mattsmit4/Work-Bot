"""
Tests for conversation logging module.

Run with: pytest tests/test_logging.py -v
"""

import pytest
import tempfile
from pathlib import Path
from ui.logging import (
    ConversationLogger,
    ConversationLog,
    get_conversation_logger
)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def logger(temp_log_file):
    """Create ConversationLogger with temp file."""
    return ConversationLogger(temp_log_file)


class TestConversationLog:
    """Test ConversationLog dataclass."""
    
    def test_creation(self):
        """Test creating ConversationLog."""
        from datetime import datetime
        
        log = ConversationLog(
            session_id="test_123",
            timestamp=datetime.now(),
            user_message="Hello",
            bot_response="Hi there!"
        )
        
        assert log.session_id == "test_123"
        assert log.user_message == "Hello"
        assert log.bot_response == "Hi there!"
        assert log.products_shown == 0
        assert log.feedback is None
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        from datetime import datetime
        
        log = ConversationLog(
            session_id="test_123",
            timestamp=datetime.now(),
            user_message="Hello",
            bot_response="Hi there!",
            intent_type="GREETING",
            products_shown=5,
            feedback="positive"
        )
        
        log_dict = log.to_dict()
        
        assert log_dict['session_id'] == "test_123"
        assert log_dict['user_message'] == "Hello"
        assert log_dict['intent_type'] == "GREETING"
        assert log_dict['products_shown'] == 5
        assert log_dict['feedback'] == "positive"


class TestConversationLogger:
    """Test ConversationLogger class."""
    
    def test_creation(self, temp_log_file):
        """Test creating ConversationLogger."""
        logger = ConversationLogger(temp_log_file)
        
        assert isinstance(logger, ConversationLogger)
        assert logger.log_file.exists()
    
    def test_log_file_created_with_headers(self, temp_log_file):
        """Test that log file is created with headers."""
        logger = ConversationLogger(temp_log_file)
        
        # Read first line
        with open(temp_log_file, 'r') as f:
            first_line = f.readline().strip()
        
        assert 'session_id' in first_line
        assert 'timestamp' in first_line
        assert 'user_message' in first_line
        assert 'bot_response' in first_line


class TestLogging:
    """Test logging conversations."""
    
    def test_log_conversation_basic(self, logger):
        """Test logging a basic conversation."""
        log = logger.log_conversation(
            session_id="session_1",
            user_message="Test message",
            bot_response="Test response"
        )
        
        assert isinstance(log, ConversationLog)
        assert log.session_id == "session_1"
        assert log.user_message == "Test message"
        assert log.bot_response == "Test response"
    
    def test_log_conversation_with_all_fields(self, logger):
        """Test logging conversation with all fields."""
        log = logger.log_conversation(
            session_id="session_1",
            user_message="Show me cables",
            bot_response="Here are 5 cables...",
            intent_type="NEW_SEARCH",
            products_shown=5,
            feedback="positive",
            metadata={'user_id': 'user_123'}
        )
        
        assert log.intent_type == "NEW_SEARCH"
        assert log.products_shown == 5
        assert log.feedback == "positive"
        assert log.metadata == {'user_id': 'user_123'}
    
    def test_log_feedback(self, logger):
        """Test logging user feedback."""
        logger.log_feedback(
            session_id="session_1",
            feedback="positive",
            message="Very helpful!"
        )
        
        conversations = logger.get_conversations()
        assert len(conversations) == 1
        assert conversations[0]['feedback'] == "positive"


class TestRetrieving:
    """Test retrieving conversations."""
    
    def test_get_conversations_empty(self, logger):
        """Test getting conversations from empty log."""
        conversations = logger.get_conversations()
        assert conversations == []
    
    def test_get_conversations_all(self, logger):
        """Test getting all conversations."""
        logger.log_conversation("s1", "msg1", "resp1")
        logger.log_conversation("s2", "msg2", "resp2")
        logger.log_conversation("s3", "msg3", "resp3")
        
        conversations = logger.get_conversations()
        
        assert len(conversations) == 3
    
    def test_get_conversations_by_session(self, logger):
        """Test getting conversations by session ID."""
        logger.log_conversation("session_1", "msg1", "resp1")
        logger.log_conversation("session_2", "msg2", "resp2")
        logger.log_conversation("session_1", "msg3", "resp3")
        
        session_1_convs = logger.get_conversations(session_id="session_1")
        
        assert len(session_1_convs) == 2
        assert all(c['session_id'] == "session_1" for c in session_1_convs)
    
    def test_get_conversations_with_limit(self, logger):
        """Test getting conversations with limit."""
        for i in range(10):
            logger.log_conversation(f"session_{i}", f"msg_{i}", f"resp_{i}")
        
        recent = logger.get_conversations(limit=3)
        
        assert len(recent) == 3
    
    def test_get_session_conversations(self, logger):
        """Test getting session conversations."""
        logger.log_conversation("session_1", "msg1", "resp1")
        logger.log_conversation("session_1", "msg2", "resp2")
        logger.log_conversation("session_2", "msg3", "resp3")
        
        session_convs = logger.get_session_conversations("session_1")
        
        assert len(session_convs) == 2


class TestStatistics:
    """Test statistics methods."""
    
    def test_get_conversation_count(self, logger):
        """Test getting conversation count."""
        assert logger.get_conversation_count() == 0
        
        logger.log_conversation("s1", "msg1", "resp1")
        logger.log_conversation("s1", "msg2", "resp2")
        
        assert logger.get_conversation_count() == 2
    
    def test_get_feedback_stats(self, logger):
        """Test getting feedback statistics."""
        logger.log_conversation("s1", "msg1", "resp1", feedback="positive")
        logger.log_conversation("s2", "msg2", "resp2", feedback="positive")
        logger.log_conversation("s3", "msg3", "resp3", feedback="negative")
        logger.log_conversation("s4", "msg4", "resp4")  # No feedback
        
        stats = logger.get_feedback_stats()
        
        assert stats['positive'] == 2
        assert stats['negative'] == 1
        assert stats['total'] == 3
    
    def test_get_sessions(self, logger):
        """Test getting list of sessions."""
        logger.log_conversation("session_1", "msg1", "resp1")
        logger.log_conversation("session_2", "msg2", "resp2")
        logger.log_conversation("session_1", "msg3", "resp3")
        
        sessions = logger.get_sessions()
        
        assert len(sessions) == 2
        assert "session_1" in sessions
        assert "session_2" in sessions
    
    def test_get_session_stats(self, logger):
        """Test getting session statistics."""
        logger.log_conversation("s1", "msg1", "resp1", products_shown=3)
        logger.log_conversation("s1", "msg2", "resp2", products_shown=5)
        logger.log_conversation("s1", "msg3", "resp3", feedback="positive")
        
        stats = logger.get_session_stats("s1")
        
        assert stats['message_count'] == 3
        assert stats['products_shown'] == 8
        assert stats['feedback'] == "positive"
    
    def test_get_session_stats_empty(self, logger):
        """Test getting stats for non-existent session."""
        stats = logger.get_session_stats("nonexistent")
        
        assert stats['message_count'] == 0
        assert stats['products_shown'] == 0
        assert stats['feedback'] is None


class TestUtilities:
    """Test utility methods."""
    
    def test_clear_logs(self, logger):
        """Test clearing all logs."""
        logger.log_conversation("s1", "msg1", "resp1")
        logger.log_conversation("s2", "msg2", "resp2")
        
        assert logger.get_conversation_count() == 2
        
        logger.clear_logs()
        
        assert logger.get_conversation_count() == 0
    
    def test_export_to_dict(self, logger):
        """Test exporting to dictionary."""
        logger.log_conversation("s1", "msg1", "resp1")
        logger.log_conversation("s2", "msg2", "resp2")
        
        data = logger.export_to_dict()
        
        assert len(data) == 2
        assert isinstance(data, list)
        assert all(isinstance(item, dict) for item in data)


class TestSingletonAccess:
    """Test singleton accessor."""
    
    def test_get_conversation_logger(self, temp_log_file):
        """Test getting logger singleton."""
        logger = get_conversation_logger(log_file=temp_log_file, reset=True)
        
        assert isinstance(logger, ConversationLogger)
    
    def test_singleton_same_instance(self, temp_log_file):
        """Test that singleton returns same instance."""
        logger1 = get_conversation_logger(log_file=temp_log_file, reset=True)
        logger2 = get_conversation_logger()
        
        assert logger1 is logger2
    
    def test_singleton_reset(self, temp_log_file):
        """Test resetting singleton."""
        logger1 = get_conversation_logger(log_file=temp_log_file, reset=True)
        logger1.log_conversation("s1", "msg1", "resp1")
        
        logger2 = get_conversation_logger(reset=True)
        
        # Should be a new instance
        assert logger2 is not logger1


# Run tests with: pytest tests/test_logging.py -v