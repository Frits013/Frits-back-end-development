#### Import all orchestration dependencies
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Union
import logging
from datetime import datetime, timezone
import uuid
from supabase._async.client import AsyncClient as AsyncSupabase

# Import pydantic-graph components
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

#### Import agent workflow functions
from .Meta_Agent.internal_logic_MA import MetaAgent_workflow  
from .Reviewer_Agent.internal_logic_RA import ReviewerAgent_workflow
from .Update_Agent.internal_logic_UA import UpdateAgent_workflow
from .Writer_Agent.internal_logic_WA import WriterAgent_workflow
from .classes import MultiAgentDeps, MultiAgentState, ChatMessage


########################################################################
# Fetching functions moved here to populate the GraphRunContext state.
########################################################################

async def fetch_user_profile(supabase_client: AsyncSupabase, user_id: str) -> dict:
    try:
        # 1) Fetch the user row (including the company_id)
        user_resp = await supabase_client.table("users") \
            .select(
                "user_description",
                "company_id",
                "distilled_company_AIR_info",
                "distilled_user_AIR_info",
                "TTS_flag"
            ) \
            .eq("user_id", user_id) \
            .execute()

        if not user_resp.data:
            # No user found – return defaults
            return {
                "user_description": "Unknown",
                "company_description": "none",
                "distilled_company_AIR_info": "none",
                "distilled_user_AIR_info": "none",
                "TTS_flag": 0
            }

        user = user_resp.data[0]
        company_id = user.get("company_id")

        # 2) Fetch the company_description using that company_id
        company_description = "none"
        if company_id:
            comp_resp = await supabase_client.table("companies") \
                .select("company_description") \
                .eq("company_id", company_id) \
                .single() \
                .execute()
            if comp_resp.data and "company_description" in comp_resp.data:
                company_description = comp_resp.data["company_description"]

        # 3) Build and return the merged profile
        return {
            "user_description": user.get("user_description", "Unknown"),
            "company_description": company_description,
            "distilled_company_AIR_info": user.get("distilled_company_AIR_info", "none"),
            "distilled_user_AIR_info": user.get("distilled_user_AIR_info", "none"),
            "TTS_flag": user.get("TTS_flag", 0)
        }

    except Exception as e:
        logging.error("Error fetching user profile: %s", e)
        return {
            "user_description": "Unknown",
            "company_description": "none",
            "distilled_company_AIR_info": "none",
            "distilled_user_AIR_info": "none",
            "TTS_flag": 0
        }

async def fetch_conversation_history(supabase_client: AsyncSupabase, session_id: str, limit: int = 30) -> tuple[dict[str, ChatMessage], dict[str, ChatMessage]]:
    """
    Fetch conversation history from Supabase for a given session_id.
    Only includes messages with role = 'writer' or 'User',
    sorted from oldest to newest, limited to the most recent `limit` messages.
    Also fetches the last system message for the session.
    Returns a tuple of (conversation_history, latest_phase_prompt).
    """
    try:
        # 1. Query messages for this session, only Frits or User
        #    sorted ascending by created_at (oldest first).
        response = await supabase_client.table("chat_messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .in_("role", ["writer", "user"]) \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()

        # 2. Dictionary to hold messages keyed by message_id
        conversation_history: dict[str, ChatMessage] = {}

        # 3. Iterate through the messages (already oldest -> newest)
        for msg in (response.data or []):
            created_at_str = msg.get("created_at")

            # Parse created_at if it's a string, fallback to "now" on error
            if isinstance(created_at_str, str):
                try:
                    # handle 'Z' by replacing with '+00:00' so fromisoformat works
                    created_at = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                except Exception as parse_err:
                    logging.error("Invalid created_at format: %s", created_at_str)
                    created_at = datetime.now(timezone.utc)
            else:
                created_at = datetime.now(timezone.utc)

            # 4. Preserve existing message_id if present, otherwise generate new
            msg_id = msg.get("message_id", str(uuid.uuid4()))

            chat_msg = ChatMessage(
                message_id=msg_id,
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                created_at=created_at
            )

            # 5. Insert into dict, keyed by the message_id
            conversation_history[msg_id] = chat_msg

        # 6. Fetch the last system message for this session
        system_response = await supabase_client.table("chat_messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .eq("role", "system") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        latest_phase_prompt: dict[str, ChatMessage] = {}
        
        # 7. Process the system message if found
        if system_response.data and len(system_response.data) > 0:
            system_msg = system_response.data[0]
            created_at_str = system_msg.get("created_at")

            # Parse created_at if it's a string, fallback to "now" on error
            if isinstance(created_at_str, str):
                try:
                    created_at = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                except Exception as parse_err:
                    logging.error("Invalid created_at format: %s", created_at_str)
                    created_at = datetime.now(timezone.utc)
            else:
                created_at = datetime.now(timezone.utc)

            msg_id = system_msg.get("message_id", str(uuid.uuid4()))

            system_chat_msg = ChatMessage(
                message_id=msg_id,
                role=system_msg.get("role", ""),
                content=system_msg.get("content", ""),
                created_at=created_at
            )

            latest_phase_prompt[msg_id] = system_chat_msg

        return conversation_history, latest_phase_prompt

    except Exception as e:
        logging.error(f"Failed to fetch conversation history: {e}")
        raise


async def fetch_message_by_id(supabase_client, message_id: str) -> ChatMessage | None:
    """
    Fetch a single chat message from Supabase using its message_id.
    
    Args:
        supabase_client: The Supabase client instance.
        message_id (str): The unique identifier of the message.
    
    Returns:
        A ChatMessage object if a matching record is found; otherwise, None.
    """
    try:
        # Query the 'chat_messages' table for a record with the given message_id.
        response = await supabase_client.table("chat_messages") \
            .select("*") \
            .eq("message_id", message_id) \
            .execute()

        # Ensure that the response contains data.
        if response.data and len(response.data) > 0:
            record = response.data[0]
            
            # Process the created_at field. It might be a string, so we try to parse it.
            created_at_raw = record.get("created_at")
            if isinstance(created_at_raw, str):
                try:
                    # Replace 'Z' with '+00:00' for proper ISO format parsing.
                    created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
                except Exception:
                    created_at = datetime.now(timezone.utc)
            elif isinstance(created_at_raw, datetime):
                created_at = created_at_raw
            else:
                created_at = datetime.now(timezone.utc)

            # Create a ChatMessage object using the fields from the record.
            chat_message = ChatMessage(
                message_id=record.get("message_id", str(uuid.uuid4())),
                role=record.get("role", ""),
                content=record.get("content", ""),
                created_at=created_at
            )
            return chat_message
        else:
            # No record found for the given message_id.
            return None

    except Exception as e:
        # Optionally, log the error.
        print(f"Error fetching message by id {message_id}: {e}")
        raise



########################################################################
# Create the GraphRunContext, fetching and setting up the state.
########################################################################

async def create_context(clients, user_id: str, payload) -> GraphRunContext[MultiAgentState, MultiAgentDeps]:
    
    # Fetch additional data for the context.
    user_profile = await fetch_user_profile(clients.supabase_client, user_id)
    conversation_history, latest_phase_prompt = await fetch_conversation_history(clients.supabase_client, payload.session_id)
    latest_user_message = await fetch_message_by_id(clients.supabase_client, payload.message_id)

    state = MultiAgentState(
        internalconversation = {latest_user_message.message_id: latest_user_message},
        latest_phase_prompt = latest_phase_prompt,
        MA_response = {},
        reviewer_response = {},
        reviewer_approval = False,
        session_finished= False,
        writer_response={},
        new_company_info={},
        new_user_AIR_info={}, 
        )


    deps = MultiAgentDeps(
          #### Add agent instances
        meta_agent=clients.meta_agent,
        reviewer_agent=clients.reviewer_agent,
        writer_agent=clients.writer_agent,
        update_agent=clients.update_agent,

        #### other dependencies from supabase needed to run multi agent
        user_id=user_id,
        session_id=payload.session_id,
        user_message=latest_user_message,
        user_profile=user_profile,
        conversation_history=conversation_history,
    )

    # Set the dependency container as the deps.
    return GraphRunContext(state=state, deps=deps)

########################################################################
# Define the nodes for the multi-agent workflow.
########################################################################
#@dataclass
#class UpdateAgentNode(BaseNode[MultiAgentState, MultiAgentDeps]): 
#    ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]

#    async def run(self, graph_ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]) -> MetaAgentNode:
#            self.ctx = await UpdateAgent_workflow(graph_ctx)

            # Pass to the MetaAgentNode.
 #           return MetaAgentNode(self.ctx)


@dataclass
class UpdateAndMetaAgentNode(BaseNode[MultiAgentState, MultiAgentDeps]):
    ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]

    async def run(self, graph_ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]) -> ReviewerAgentNode:
        # fire both workflows at once
        update_task = UpdateAgent_workflow(graph_ctx)
        meta_task   = MetaAgent_workflow(graph_ctx)

        # wait for both to complete
        await asyncio.gather(update_task, meta_task)

        # (both workflows have mutated graph_ctx in place;
        #  you can pick either returned ctx or just use graph_ctx itself)
        self.ctx = graph_ctx

        # pass the unified context on to the reviewer
        return ReviewerAgentNode(self.ctx)
    
@dataclass
class MetaAgentNode(BaseNode[MultiAgentState, MultiAgentDeps]):
    ctx: GraphRunContext[MultiAgentState, MultiAgentDeps] 

    async def run(self, graph_ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]) -> ReviewerAgentNode:
        self.ctx = await MetaAgent_workflow(graph_ctx)

        # Pass to the ReviewerAgentNode.
        return ReviewerAgentNode(self.ctx)


@dataclass
class ReviewerAgentNode(BaseNode[MultiAgentState, MultiAgentDeps]):
    ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]

    async def run(self, graph_ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]) -> Union[MetaAgentNode, WriterAgentNode]:
        # Run the reviewer workflow using the reviewer agent.
        self.ctx = await ReviewerAgent_workflow(graph_ctx)
        
        # Decide the next node based on the reviewer_approval flag.
        if self.ctx.state.reviewer_approval == 1:
            logging.info("Reviewer approved the response. Routing to AudioNode.")
            return WriterAgentNode(self.ctx)
        else:
            logging.info("Reviewer did not approve the response. Routing back to MainAgentNode.")
            return MetaAgentNode(self.ctx)

@dataclass
class WriterAgentNode(BaseNode[MultiAgentState, MultiAgentDeps]):
    ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]

    async def run(self, graph_ctx: GraphRunContext[MultiAgentState, MultiAgentDeps]) -> End:
        # Run the reviewer workflow using the reviewer agent.
        self.ctx = await WriterAgent_workflow(graph_ctx)
        
        # Decide the next node based on the reviewer_approval flag.
        if self.ctx.deps.user_profile["TTS_flag"] == 1:
            logging.info("User put TTS_flag as 1, routing to audio agent")
            return End(self.ctx)
        else:
            logging.info("TTS_flag was 0 so answer as text")
            return End(self.ctx)



########################################################################
# Build the graph.
########################################################################
multi_agent_graph = Graph(nodes=[UpdateAndMetaAgentNode, MetaAgentNode, ReviewerAgentNode, WriterAgentNode])

########################################################################
# Runner function to start the orchestration workflow.
########################################################################

async def run_multi_agent_workflow(clients, user_id: str, payload) -> GraphRunContext[MultiAgentState, MultiAgentDeps]:
    # Create GraphRunContext including dependencies.
    initial_ctx = await create_context(clients, user_id, payload)
    
    # Run the graph using the initial node.
    end_node = await multi_agent_graph.run(start_node=UpdateAndMetaAgentNode(initial_ctx), state=initial_ctx.state, deps=initial_ctx.deps) ### check dit nog
    
    return end_node
