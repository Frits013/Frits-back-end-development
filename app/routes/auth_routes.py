import logging
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from ..auth import validate_supabase_token, create_fastapi_token

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
router = APIRouter()

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/token", response_model=TokenResponse, tags=["authentication"])
def get_fastapi_token(authorization: str = Header(None)):
    logger.debug("Received token request via header")
    if not authorization or not authorization.startswith("Bearer "):
        logger.error("Missing or invalid authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    # Extract the token from the header.
    supabase_token = authorization.removeprefix("Bearer ").strip()
    logger.debug("Validating Supabase token")
    user_info = validate_supabase_token(supabase_token)

    user_id = user_info["user_id"]
    role = user_info["role"]

    logger.debug(f"Generating FastAPI token for user_id={user_id}, role={role}")
    fastapi_token = create_fastapi_token(user_id, role)
    logger.info(f"Generated FastAPI token for user_id={user_id}, role={role}")

    return TokenResponse(access_token=fastapi_token)
