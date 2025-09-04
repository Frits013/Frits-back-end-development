##### for ChatMessage class
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from pydantic_ai import Agent
from pydantic import BaseModel

################# Chatmessage class
@dataclass
class ChatMessage:
    # Automatically generate a unique message ID on creation
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = ""     
    content: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class CompanyInfoMessage:
    info_id:str = field(default_factory=lambda: str(uuid.uuid4()))
    content_dict: dict = field(default_factory=dict)
    content_str: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class UserInfoMessage:
    info_id:str = field(default_factory=lambda: str(uuid.uuid4()))
    content_dict: dict = field(default_factory=dict)
    content_str: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MultiAgentState:
    # CONVERSATION BETWEEN AGENTS
    internalconversation: dict[str, ChatMessage] = field(default_factory=dict)
    latest_phase_prompt: dict[str, ChatMessage] = field(default_factory=dict)
    
    # META-AGENT
    MA_response: dict[str, ChatMessage] = field(default_factory=dict)

    # REVIEWER
    reviewer_response: dict[str, ChatMessage] = field(default_factory=dict)
    # When this flag switches to True during run, it stops the loop Frits <--> Reviewer
    reviewer_approval: bool = False
    session_finished: bool = False
    
    # WRITER AGENT
    writer_response: Optional[ChatMessage] = None

    # GRADING AGENT
    new_company_info: dict[str, CompanyInfoMessage] = field(default_factory=dict)
    new_user_AIR_info: dict[str, UserInfoMessage] = field(default_factory=dict)


@dataclass
class MultiAgentDeps:
    # Agents used in the multi-agent system
    meta_agent: Agent
    reviewer_agent: Agent
    writer_agent: Agent
    update_agent: Agent

    # Filled by front-end and supabase
    user_id: str
    user_profile: Optional[dict] = None

    # Session information
    session_id: str = ""
    user_message: ChatMessage = field(default_factory=dict)
    conversation_history: dict[str, ChatMessage] = field(default_factory=dict)



####### individual agent run dependency classes
@dataclass
class review_agent_deps:  
    RAG_tool_URL: str
    RAG_tool_KEY: str
    RAG_input: Optional[str] = None
    RAG_response: str = ""
    

# Define a schema for incoming and outgoing messages.
class InputMessage(BaseModel):
    message_id: str 
    session_id: str 
    

class OutputMessage(BaseModel):
    response: str
    session_id: str