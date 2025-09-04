import logging
from fastapi import APIRouter, Depends, Request
from ..auth import get_current_user
from ..orchestration import run_multi_agent_workflow
from pydantic_ai.exceptions import ModelHTTPError
from supabase._async.client import AsyncClient as AsyncSupabase
import asyncio
import threading
from ..classes import MultiAgentState, InputMessage, OutputMessage


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()


async def store_chat_messages(supabase_client: AsyncSupabase, run_info: MultiAgentState, user_id: str, session_id: str):
    chat_records = []

    # From meta agent responses
    for msg in run_info.MA_response.values():
        chat_records.append({
            "message_id": msg.message_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat()
        })

    # From reviewer responses
    for msg in run_info.reviewer_response.values():
        chat_records.append({
            "message_id": msg.message_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat()
        })

    # Writer response
    if run_info.writer_response:
        msg = run_info.writer_response
        chat_records.append({
            "message_id": msg.message_id,
            "user_id": user_id,
            "session_id": session_id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat()
        })

    if chat_records:
        # Bulk insert into chat_messages table
        await supabase_client.from_("chat_messages").insert(chat_records).execute()


async def store_info_messages(supabase_client: AsyncSupabase, run_info: MultiAgentState, user_id: str, payload):
    info_records = []

    async def update_user_info(column: str, new_content: str):
        response = await supabase_client.from_("users").select(column).eq("user_id", user_id).execute()
        current_value = ""
        if response.data and len(response.data) > 0:
            current_value = response.data[0].get(column) or ""
        updated_value = f"{current_value}\n{new_content}" if current_value else new_content
        await supabase_client.from_("users").update({column: updated_value}).eq("user_id", user_id).execute()

    # Process new_company_info messages
    for info in run_info.new_company_info.values():
        info_records.append({
            "info_id": info.info_id,
            "message_id": payload.message_id,  # This should reference a chat message that now exists
            "category": "company",
            "content_dict": info.content_dict,
            "content_str": info.content_str,
            "created_at": info.created_at.isoformat()
        })
        await update_user_info("distilled_company_AIR_info", info.content_str)

    # Process new_user_AIR_info messages
    for info in run_info.new_user_AIR_info.values():
        info_records.append({
            "info_id": info.info_id,
            "message_id": payload.message_id,
            "category": "user_air",
            "content_dict": info.content_dict,
            "content_str": info.content_str,
            "created_at": info.created_at.isoformat()
        })
        await update_user_info("distilled_user_AIR_info", info.content_str)

    if info_records:
        await supabase_client.from_("info_messages").insert(info_records).execute()

async def update_session_info(supabase_client, finalstate, session_id: str) -> None:
    """
    Update the 'finished' boolean in the 'chat_sessions' table for the given session.
    """
    # Extract the finished flag from finalstate
    finished_flag = getattr(finalstate, "session_finished", False)

    try:
        # Update the chat_sessions table
        response = await supabase_client.table("chat_sessions") \
            .update({"finished": finished_flag}) \
            .eq("id", session_id) \
            .execute()

        # Check for errors in the response
        if hasattr(response, "error") and response.error:
            logger.error(
                "Failed to update chat_sessions for session_id=%s: %s",
                session_id,
                response.error
            )
        else:
            logger.info(
                "chat_sessions.finished set to %s for session_id=%s",
                finished_flag,
                session_id
            )
    except Exception as e:
        logger.exception(
            "Exception when updating chat_sessions for session_id=%s: %s",
            session_id,
            e
        )
        

@router.post("/send_message", response_model=dict, tags=["chat"])
async def send_message(request: Request, payload: InputMessage, user: dict = Depends(get_current_user)):
    # For diagnostics: log current thread and event loop details.
    current_thread = threading.current_thread().name
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    logger.info(f"send_message called on thread: {current_thread}, current_loop: {current_loop}")

    user_id = user["user_id"]
    role = user["role"]

    # Check user permissions.
    if role not in ["admin", "authenticated"]:
        logger.warning(f"User {user_id} with role {role} attempted unauthorized access.")
        return {
            "error": True,
            "response": "Insufficient permissions.",
            "session_id": payload.session_id
        }

    # Retrieve the clients container from app.state.
    clients = request.app.state.clients

    Fritsmessage = None
    finalstate = None
    error_occurred = False

    try:
        logger.info(f"Calling multi-agent workflow for user_id={user['user_id']}, session_id={payload.session_id}")
        result = await run_multi_agent_workflow(clients, user_id, payload)
        if isinstance(result, tuple):
            result = result[0]
        finalstate = result.state
        Fritsmessage = finalstate.writer_response.content
        logger.info("Multi-agent workflow completed successfully.")
        
    except ModelHTTPError as err:
        # Use the "error" key if it exists, otherwise use err.body directly.
        error_data = err.body.get("error") or err.body
        if error_data.get("code") == "content_filter":
            Fritsmessage = "Sorry, this prompt was filtered due to our content management policy. Please modify your input and try again."
            error_occurred = True
            logger.error("Content policy violation detected. Returning error: %s", Fritsmessage)
        else:
            Fritsmessage = "An error occurred while processing your request."
            error_occurred = True
            logger.error("ModelHTTPError in multi-agent workflow: %s", err, exc_info=True)
    except Exception as e:
        logger.error("Error in multi-agent workflow: %s", e, exc_info=True)
        Fritsmessage = "An error occurred while processing your request."
        error_occurred = True

    if finalstate:
        await store_chat_messages(clients.supabase_client, finalstate, user_id, payload.session_id)
        await store_info_messages(clients.supabase_client, finalstate, user_id, payload)
        
        if finalstate.session_finished:
            logger.info("Session finished; updating session_info in Supabase")
            await update_session_info(clients.supabase_client, finalstate, payload.session_id)
    
    logger.info(f"Processed message for user_id={user_id}, session_id={payload.session_id}")
    logger.info("Returning error: %s, message: %s", error_occurred, Fritsmessage)

    return {
        "error": error_occurred,
        "response": Fritsmessage,
        "session_id": payload.session_id
    }