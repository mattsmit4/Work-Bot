"""
Session state management for ST-Bot.

Handles conversation history, product context, user preferences,
and Streamlit session persistence for guidance/question flows.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from core.context import (
    Product, ConversationContext, IntentType,
    PendingGuidance, GuidancePhase, PendingQuestion, PendingQuestionType
)


@dataclass
class Message:
    """
    Represents a single message in the conversation.
    
    Attributes:
        role: 'user' or 'assistant'
        content: Message text
        timestamp: When message was created
        metadata: Additional message data
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionState:
    """
    Manages session state for a chatbot conversation.
    
    Tracks:
    - Conversation history (all messages)
    - Product context (last search results)
    - User preferences
    - Session metadata
    
    Example:
        state = SessionState()
        state.add_message("user", "I need a USB-C cable")
        state.add_message("assistant", "Here are some options...")
        
        # Store search results
        state.set_product_context([product1, product2])
        
        # Get conversation history
        history = state.get_conversation_history()
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session state.
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or self._generate_session_id()
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Conversation tracking
        self._messages: List[Message] = []
        self._conversation_context = ConversationContext()
        
        # User preferences
        self._preferences: Dict[str, Any] = {}
        
        # Metadata
        self._metadata: Dict[str, Any] = {}
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message text
            metadata: Optional message metadata
            
        Returns:
            Created Message object
            
        Example:
            >>> state.add_message("user", "Show me HDMI cables")
            >>> state.add_message("assistant", "Here are 5 options...")
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self._messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_conversation_history(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None
    ) -> List[Message]:
        """
        Get conversation history.
        
        Args:
            limit: Maximum number of messages to return (most recent)
            role: Filter by role ('user' or 'assistant')
            
        Returns:
            List of Message objects
            
        Example:
            >>> # Get last 10 messages
            >>> recent = state.get_conversation_history(limit=10)
            >>> # Get only user messages
            >>> user_msgs = state.get_conversation_history(role='user')
        """
        messages = self._messages
        
        # Filter by role
        if role:
            messages = [m for m in messages if m.role == role]
        
        # Limit results
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_last_message(self, role: Optional[str] = None) -> Optional[Message]:
        """
        Get the last message.
        
        Args:
            role: Optional role filter
            
        Returns:
            Last Message or None
            
        Example:
            >>> last_user = state.get_last_message(role='user')
        """
        messages = self.get_conversation_history(role=role)
        return messages[-1] if messages else None
    
    def clear_messages(self):
        """
        Clear all conversation messages.
        
        Example:
            >>> state.clear_messages()
        """
        self._messages = []
        self.updated_at = datetime.now()
    
    def set_product_context(
        self,
        products: List[Product],
        intent_type: Optional[IntentType] = None
    ):
        """
        Set product context from search results.
        
        Args:
            products: List of Product objects
            intent_type: Optional intent type
            
        Example:
            >>> state.set_product_context([product1, product2])
        """
        if len(products) == 1:
            self._conversation_context.set_single_product(products[0])
        else:
            self._conversation_context.set_multi_products(products)
        self.updated_at = datetime.now()
    
    def get_product_context(self) -> ConversationContext:
        """
        Get current product context.
        
        Returns:
            ConversationContext object
            
        Example:
            >>> context = state.get_product_context()
            >>> products = context.last_products
        """
        return self._conversation_context
    
    def clear_product_context(self):
        """
        Clear product context.
        
        Example:
            >>> state.clear_product_context()
        """
        self._conversation_context.clear_products()
        self.updated_at = datetime.now()
    
    def set_preference(self, key: str, value: Any):
        """
        Set a user preference.
        
        Args:
            key: Preference key
            value: Preference value
            
        Example:
            >>> state.set_preference("display_count", 5)
            >>> state.set_preference("show_prices", True)
        """
        self._preferences[key] = value
        self.updated_at = datetime.now()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if not found
            
        Returns:
            Preference value or default
            
        Example:
            >>> count = state.get_preference("display_count", 10)
        """
        return self._preferences.get(key, default)
    
    def clear_preferences(self):
        """
        Clear all user preferences.
        
        Example:
            >>> state.clear_preferences()
        """
        self._preferences = {}
        self.updated_at = datetime.now()
    
    def set_metadata(self, key: str, value: Any):
        """
        Set session metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Example:
            >>> state.set_metadata("user_id", "user123")
        """
        self._metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get session metadata.
        
        Args:
            key: Metadata key
            default: Default value if not found
            
        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)
    
    def get_message_count(self) -> int:
        """
        Get total message count.
        
        Returns:
            Number of messages
            
        Example:
            >>> count = state.get_message_count()
        """
        return len(self._messages)
    
    def get_session_duration(self) -> float:
        """
        Get session duration in seconds.
        
        Returns:
            Duration in seconds
            
        Example:
            >>> duration = state.get_session_duration()
        """
        return (self.updated_at - self.created_at).total_seconds()
    
    def reset(self):
        """
        Reset session to initial state.
        
        Clears:
        - All messages
        - Product context
        - Preferences
        - Metadata
        
        Keeps:
        - Session ID
        - Created timestamp
        
        Example:
            >>> state.reset()
        """
        self._messages = []
        self._conversation_context = ConversationContext()
        self._preferences = {}
        self._metadata = {}
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export session state to dictionary.
        
        Returns:
            Dictionary representation of state
            
        Example:
            >>> state_dict = state.to_dict()
        """
        # Calculate product count
        product_count = 0
        if self._conversation_context.current_products:
            product_count = len(self._conversation_context.current_products)
        elif self._conversation_context.last_product:
            product_count = 1
        
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self._messages),
            'messages': [
                {
                    'role': m.role,
                    'content': m.content,
                    'timestamp': m.timestamp.isoformat(),
                    'metadata': m.metadata
                }
                for m in self._messages
            ],
            'product_count': product_count,
            'preferences': self._preferences,
            'metadata': self._metadata
        }


# Singleton for easy access
_session_state: Optional[SessionState] = None


def get_session_state(reset: bool = False) -> SessionState:
    """
    Get the global session state instance.
    
    Args:
        reset: If True, create a new session state
        
    Returns:
        SessionState instance
        
    Example:
        >>> state = get_session_state()
        >>> state.add_message("user", "Hello")
    """
    global _session_state
    
    if reset or _session_state is None:
        _session_state = SessionState()
    
    return _session_state


# =============================================================================
# STREAMLIT SESSION PERSISTENCE
# =============================================================================
# These functions handle persistence of dataclasses with Enums across Streamlit
# reruns. Streamlit can't serialize dataclasses with Enums properly, so we
# convert to/from plain dicts.


def save_guidance_to_session(context: ConversationContext, st_session_state: Any) -> None:
    """
    Save pending guidance to Streamlit session state as a simple dict.

    Streamlit can't serialize dataclasses with Enums properly, so we
    convert to a plain dict for persistence.

    Args:
        context: ConversationContext with pending_guidance
        st_session_state: Streamlit's st.session_state object
    """
    if context.pending_guidance:
        pg = context.pending_guidance
        data = {
            'setup_type': pg.setup_type,
            'monitor_count': pg.monitor_count,
            'phase': pg.phase.value,  # Convert Enum to string
            'computer_ports': pg.computer_ports,
            'computer_port_counts': pg.computer_port_counts,
            'monitor_inputs': pg.monitor_inputs,
            'preference': pg.preference,
            # KVM-specific fields
            'kvm_port_count': getattr(pg, 'kvm_port_count', None),
            'kvm_video_type': getattr(pg, 'kvm_video_type', None),
            'kvm_usb_switching': getattr(pg, 'kvm_usb_switching', None),
            # Dock-specific fields
            'dock_monitor_count': getattr(pg, 'dock_monitor_count', None),
            'dock_power_delivery': getattr(pg, 'dock_power_delivery', None),
            'dock_ethernet': getattr(pg, 'dock_ethernet', None),
            # Cable-specific fields
            'cable_length': getattr(pg, 'cable_length', None),
        }
        st_session_state.pending_guidance_data = data
    else:
        st_session_state.pending_guidance_data = None


def load_guidance_from_session(context: ConversationContext, st_session_state: Any) -> None:
    """
    Load pending guidance from Streamlit session state back into context.

    Reconstructs the PendingGuidance dataclass from the stored dict.

    Args:
        context: ConversationContext to populate
        st_session_state: Streamlit's st.session_state object
    """
    data = getattr(st_session_state, 'pending_guidance_data', None)

    if data:
        context.pending_guidance = PendingGuidance(
            setup_type=data['setup_type'],
            monitor_count=data['monitor_count'],
            phase=GuidancePhase(data['phase']),  # Convert string back to Enum
            computer_ports=data['computer_ports'],
            computer_port_counts=data['computer_port_counts'],
            monitor_inputs=data['monitor_inputs'],
            preference=data['preference'],
        )
        # Restore optional fields if they exist
        if data.get('kvm_port_count') is not None:
            context.pending_guidance.kvm_port_count = data['kvm_port_count']
        if data.get('kvm_video_type') is not None:
            context.pending_guidance.kvm_video_type = data['kvm_video_type']
        if data.get('kvm_usb_switching') is not None:
            context.pending_guidance.kvm_usb_switching = data['kvm_usb_switching']
        if data.get('dock_monitor_count') is not None:
            context.pending_guidance.dock_monitor_count = data['dock_monitor_count']
        if data.get('dock_power_delivery') is not None:
            context.pending_guidance.dock_power_delivery = data['dock_power_delivery']
        if data.get('dock_ethernet') is not None:
            context.pending_guidance.dock_ethernet = data['dock_ethernet']
        if data.get('cable_length') is not None:
            context.pending_guidance.cable_length = data['cable_length']
    else:
        context.pending_guidance = None


def save_pending_question_to_session(context: ConversationContext, st_session_state: Any) -> None:
    """
    Save pending question to Streamlit session state as a simple dict.

    Args:
        context: ConversationContext with pending_question
        st_session_state: Streamlit's st.session_state object
    """
    if context.pending_question:
        pq = context.pending_question
        data = {
            'question_type': pq.question_type.value,  # Convert Enum to string
            'context_data': pq.context_data,
        }
        st_session_state.pending_question_data = data
    else:
        st_session_state.pending_question_data = None


def load_pending_question_from_session(context: ConversationContext, st_session_state: Any) -> None:
    """
    Load pending question from Streamlit session state back into context.

    Args:
        context: ConversationContext to populate
        st_session_state: Streamlit's st.session_state object
    """
    data = getattr(st_session_state, 'pending_question_data', None)

    if data:
        context.pending_question = PendingQuestion(
            question_type=PendingQuestionType(data['question_type']),
            context_data=data['context_data'],
        )
    else:
        context.pending_question = None