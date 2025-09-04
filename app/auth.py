import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from fastapi import Header, HTTPException, status
from jose import jwt as jose_jwt  # Using python-jose to handle JWTs

# Configure logging to capture debug messages.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables (adjust override as needed).
load_dotenv(override=False)

# JWT configuration values
FASTAPI_JWT_SECRET = os.getenv("FASTAPI_JWT_SECRET")
FASTAPI_JWT_ISSUER = os.getenv("FASTAPI_JWT_ISSUER")
FASTAPI_JWT_AUDIENCE = os.getenv("FASTAPI_JWT_AUDIENCE")
FASTAPI_JWT_ALGORITHM = os.getenv("FASTAPI_JWT_ALGORITHM", "HS256")
FASTAPI_JWT_EXPIRATION_MINUTES = os.getenv("FASTAPI_JWT_EXPIRATION_MINUTES", "60")

logger.debug("JWT configuration loaded")

# Ensure that required variables are set.
missing_fastapi_vars = [
    var for var in ["FASTAPI_JWT_SECRET", "FASTAPI_JWT_ISSUER", "FASTAPI_JWT_AUDIENCE"]
    if not os.getenv(var) and locals().get(var) is None
]
if missing_fastapi_vars:
    error_msg = f"Missing FastAPI JWT environment variables: {', '.join(missing_fastapi_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Convert expiration to integer.
try:
    FASTAPI_JWT_EXPIRATION_MINUTES = int(FASTAPI_JWT_EXPIRATION_MINUTES)
    logger.debug("FASTAPI_JWT_EXPIRATION_MINUTES converted to integer")
except ValueError:
    logger.error("FASTAPI_JWT_EXPIRATION_MINUTES must be an integer.")
    raise ValueError("FASTAPI_JWT_EXPIRATION_MINUTES must be an integer.")

# Load Supabase JWT secret.
def get_supabase_secret() -> str:
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET is not set")
    return secret

def get_supabase_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise ValueError("Supabase URL not set")
    return url

def validate_supabase_token(token: str) -> dict:
    logger.debug("Validating Supabase token")
    try:
        payload = jose_jwt.decode(token, key=get_supabase_secret(), algorithms=["HS256"], audience="authenticated")
        logger.debug("Decoded Supabase token payload")
        
        user_id = payload.get("sub")
        role = payload.get("role")

        if not user_id:
            logger.warning("Supabase token missing 'sub' claim.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload invalid: user ID missing",
            )
        if not role:
            logger.warning("Supabase token missing 'role' claim.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload invalid: role missing",
            )

        return {"user_id": user_id, "role": role}

    except jose_jwt.ExpiredSignatureError:
        logger.warning("Supabase token has expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase token has expired",
        )
    except jose_jwt.JWTError as e:
        logger.warning(f"Invalid Supabase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Supabase token",
        )

def create_fastapi_token(user_id: str, role: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(minutes=FASTAPI_JWT_EXPIRATION_MINUTES)
    to_encode = {
        "sub": user_id,
        "exp": expiration,
        "role": role,
        "iss": FASTAPI_JWT_ISSUER,
        "aud": FASTAPI_JWT_AUDIENCE,
    }
    logger.debug(f"Creating FastAPI token with payload: {{'sub': {user_id}, 'exp': {expiration}, 'role': {role}}}")
    encoded_jwt = jose_jwt.encode(
        to_encode,
        FASTAPI_JWT_SECRET,
        algorithm=FASTAPI_JWT_ALGORITHM
    )
    logger.debug("FastAPI token successfully created")
    return encoded_jwt

def decode_fastapi_token(token: str) -> dict:
    logger.debug("Decoding FastAPI token")
    try:
        payload = jose_jwt.decode(
            token,
            FASTAPI_JWT_SECRET,
            algorithms=[FASTAPI_JWT_ALGORITHM],
            issuer=FASTAPI_JWT_ISSUER,
            audience=FASTAPI_JWT_AUDIENCE,
        )
        logger.debug("FastAPI token payload decoded")
        user_id = payload.get("sub")
        role = payload.get("role")

        if not user_id or not role:
            logger.warning("FastAPI token missing required claims.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload invalid: missing required claims",
            )

        return {"user_id": user_id, "role": role}

    except jose_jwt.ExpiredSignatureError:
        logger.warning("FastAPI token has expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="FastAPI token has expired",
        )
    except jose_jwt.JWTError as e:
        logger.warning(f"Invalid FastAPI token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid FastAPI token",
        )

def get_current_user(authorization: str = Header(None)) -> dict:
    """
    Dependency that extracts the Supabase token from the Authorization header,
    decodes it, and returns the user information.
    """
    logger.debug("Extracting current user from authorization header")
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Authorization header missing or does not start with Bearer.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid token",
        )

    token = authorization.removeprefix("Bearer ").strip()
    user_info = decode_fastapi_token(token)
    logger.debug(f"User info extracted: {user_info}")
    return user_info
