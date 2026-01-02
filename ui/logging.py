"""
Conversation logging for ST-Bot.

Logs conversations to CSV for analysis and improvement.
"""

import csv
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class ConversationLog:
    """
    Represents a single conversation log entry.
    
    Attributes:
        session_id: Unique session identifier
        timestamp: When the conversation occurred
        user_message: User's message
        bot_response: Bot's response
        intent_type: Detected intent type
        products_shown: Number of products shown
        feedback: User feedback ('positive', 'negative', or None)
        metadata: Additional log data
    """
    session_id: str
    timestamp: datetime
    user_message: str
    bot_response: str
    intent_type: Optional[str] = None
    products_shown: int = 0
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV writing."""
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'user_message': self.user_message,
            'bot_response': self.bot_response,
            'intent_type': self.intent_type or '',
            'products_shown': self.products_shown,
            'feedback': self.feedback or '',
            'metadata': str(self.metadata) if self.metadata else ''
        }


class ConversationLogger:
    """
    Logs conversations to CSV file.
    
    Features:
    - Log conversations with timestamps
    - Track user feedback
    - Export/import conversation history
    - Session tracking
    
    Example:
        logger = ConversationLogger("conversations.csv")
        logger.log_conversation(
            session_id="session_123",
            user_message="I need a cable",
            bot_response="Here are some options...",
            feedback="positive"
        )
    """
    
    CSV_HEADERS = [
        'session_id',
        'timestamp',
        'user_message',
        'bot_response',
        'intent_type',
        'products_shown',
        'feedback',
        'metadata'
    ]
    
    def __init__(self, log_file: str = "conversation_logs.csv"):
        """
        Initialize conversation logger.
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = Path(log_file)
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file exists with headers."""
        # Check if file exists and has content
        file_exists = self.log_file.exists()
        file_has_content = file_exists and self.log_file.stat().st_size > 0
        
        if not file_has_content:
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                writer.writeheader()
    
    def log_conversation(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        intent_type: Optional[str] = None,
        products_shown: int = 0,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationLog:
        """
        Log a conversation to CSV.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            bot_response: Bot's response
            intent_type: Detected intent type
            products_shown: Number of products shown
            feedback: User feedback ('positive', 'negative', or None)
            metadata: Additional data to log
            
        Returns:
            ConversationLog object
            
        Example:
            >>> logger.log_conversation(
            ...     session_id="session_123",
            ...     user_message="Show me HDMI cables",
            ...     bot_response="Here are 5 HDMI cables...",
            ...     intent_type="NEW_SEARCH",
            ...     products_shown=5,
            ...     feedback="positive"
            ... )
        """
        log_entry = ConversationLog(
            session_id=session_id,
            timestamp=datetime.now(),
            user_message=user_message,
            bot_response=bot_response,
            intent_type=intent_type,
            products_shown=products_shown,
            feedback=feedback,
            metadata=metadata or {}
        )
        
        # Write to CSV
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writerow(log_entry.to_dict())
        
        return log_entry
    
    def log_feedback(
        self,
        session_id: str,
        feedback: str,
        message: Optional[str] = None
    ):
        """
        Log user feedback.
        
        Args:
            session_id: Session identifier
            feedback: 'positive' or 'negative'
            message: Optional feedback message
            
        Example:
            >>> logger.log_feedback(
            ...     session_id="session_123",
            ...     feedback="positive",
            ...     message="Very helpful!"
            ... )
        """
        metadata = {'feedback_message': message} if message else {}
        
        self.log_conversation(
            session_id=session_id,
            user_message="[FEEDBACK]",
            bot_response="",
            feedback=feedback,
            metadata=metadata
        )
    
    def get_conversations(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history from CSV.
        
        Args:
            session_id: Filter by session ID
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries
            
        Example:
            >>> # Get all conversations
            >>> all_logs = logger.get_conversations()
            >>> # Get conversations for specific session
            >>> session_logs = logger.get_conversations(session_id="session_123")
            >>> # Get last 10 conversations
            >>> recent = logger.get_conversations(limit=10)
        """
        if not self.log_file.exists():
            return []
        
        conversations = []
        
        with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by session if specified
                if session_id and row['session_id'] != session_id:
                    continue
                
                conversations.append(row)
        
        # Apply limit
        if limit:
            conversations = conversations[-limit:]
        
        return conversations
    
    def get_session_conversations(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversations for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversations for the session
            
        Example:
            >>> logs = logger.get_session_conversations("session_123")
        """
        return self.get_conversations(session_id=session_id)
    
    def get_feedback_stats(self) -> Dict[str, int]:
        """
        Get feedback statistics.
        
        Returns:
            Dictionary with feedback counts
            
        Example:
            >>> stats = logger.get_feedback_stats()
            >>> print(stats)
            {'positive': 45, 'negative': 5, 'total': 50}
        """
        conversations = self.get_conversations()
        
        positive = sum(1 for c in conversations if c.get('feedback') == 'positive')
        negative = sum(1 for c in conversations if c.get('feedback') == 'negative')
        total = len([c for c in conversations if c.get('feedback')])
        
        return {
            'positive': positive,
            'negative': negative,
            'total': total
        }
    
    def get_conversation_count(self) -> int:
        """
        Get total conversation count.
        
        Returns:
            Number of logged conversations
            
        Example:
            >>> count = logger.get_conversation_count()
        """
        return len(self.get_conversations())
    
    def clear_logs(self):
        """
        Clear all conversation logs.
        
        Warning: This deletes all logged conversations!
        
        Example:
            >>> logger.clear_logs()
        """
        if self.log_file.exists():
            self.log_file.unlink()
        self._ensure_log_file()
    
    def export_to_dict(self) -> List[Dict[str, Any]]:
        """
        Export all conversations to list of dictionaries.
        
        Returns:
            List of all conversations
            
        Example:
            >>> data = logger.export_to_dict()
        """
        return self.get_conversations()
    
    def get_sessions(self) -> List[str]:
        """
        Get list of unique session IDs.
        
        Returns:
            List of session IDs
            
        Example:
            >>> sessions = logger.get_sessions()
        """
        conversations = self.get_conversations()
        sessions = set(c['session_id'] for c in conversations)
        return sorted(sessions)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
            
        Example:
            >>> stats = logger.get_session_stats("session_123")
            >>> print(stats)
            {'message_count': 10, 'products_shown': 25, 'feedback': 'positive'}
        """
        conversations = self.get_session_conversations(session_id)
        
        if not conversations:
            return {
                'message_count': 0,
                'products_shown': 0,
                'feedback': None
            }
        
        # Calculate stats
        message_count = len(conversations)
        products_shown = sum(
            int(c.get('products_shown', 0) or 0) 
            for c in conversations
        )
        
        # Get feedback (most recent)
        feedback = None
        for conv in reversed(conversations):
            if conv.get('feedback'):
                feedback = conv['feedback']
                break
        
        return {
            'message_count': message_count,
            'products_shown': products_shown,
            'feedback': feedback
        }


# Singleton for easy access
_conversation_logger: Optional[ConversationLogger] = None


def get_conversation_logger(
    log_file: str = "conversation_logs.csv",
    reset: bool = False
) -> ConversationLogger:
    """
    Get the conversation logger instance.
    
    Args:
        log_file: Path to CSV log file
        reset: If True, create a new logger instance
        
    Returns:
        ConversationLogger instance
        
    Example:
        >>> logger = get_conversation_logger()
        >>> logger.log_conversation(
        ...     session_id="session_123",
        ...     user_message="Hello",
        ...     bot_response="Hi! How can I help?"
        ... )
    """
    global _conversation_logger
    
    if reset or _conversation_logger is None:
        _conversation_logger = ConversationLogger(log_file)
    
    return _conversation_logger