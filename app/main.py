import os
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logfire
import logging


### to run locally:           uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
load_dotenv()  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize and tear down application resources.
    """
    from .dependencies import init_clients
    
    # Initialize clients and attach to app state
    app.state.clients = await init_clients()

    # Debug: log available client attributes after init
    logging.info(f"clients attrs: {dir(app.state.clients)}")

    try:
        yield
    finally:
        # Optional cleanup: close any async connections if supported
        clients = app.state.clients
        try:
            await clients.supabase_client.close()
        except Exception:
            pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="Chat API", lifespan=lifespan)



logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), scrubbing=False) #### move this to environent variable en scrubbing nu helemaal uit, maak filter
logfire.instrument_fastapi(app, capture_headers=True)   
logfire.instrument_httpx(capture_all=True)
logfire.instrument_pydantic_ai()


# Configure CORS immediately after app creation
origins = [
    "http://localhost:8080",  
    "https://preview--frits-conversation-portal.lovable.app",
    "http://192.168.0.104:8080",
    "http://10.58.87.216:8080",
    "http://192.168.10.44:8080",
    "http://192.168.0.102:8080"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Now define your test routes
@app.get("/")
async def read_root():
    return {"Hello": "Curious person"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Include other routers
from .routes import auth_routes, chat_routes
app.include_router(auth_routes.router, prefix="/auth")
app.include_router(chat_routes.router, prefix="/chat")


