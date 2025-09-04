# dependencies.py
import os
from dotenv import load_dotenv

from supabase._async.client import AsyncClient

from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from .classes import review_agent_deps

load_dotenv()  # Ensure env variables are loaded


class Clients:
    def __init__(
        self,
        supabase_client: AsyncClient,
        azure_client: AsyncAzureOpenAI,


        model_update: OpenAIModel,
        update_agent: Agent,
        model_meta: OpenAIModel,
        meta_agent: Agent,
        model_reviewer: OpenAIModel,
        reviewer_agent: Agent,
        model_writer: OpenAIModel,
        writer_agent: Agent,
    ):
        self.supabase_client = supabase_client
        self.azure_client = azure_client

        self.model_update = model_update
        self.update_agent = update_agent

        self.model_meta = model_meta
        self.meta_agent = meta_agent

        self.model_reviewer = model_reviewer
        self.reviewer_agent = reviewer_agent

        self.model_writer = model_writer
        self.writer_agent = writer_agent


async def init_clients() -> Clients:
    # --- Supabase client ---
    supabase = AsyncClient(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SERVICE_KEY"),
    )

    # --- Azure OpenAI client ---
    azure = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_RESOURCE_API_VERSION"),
        api_key=os.getenv("AZURE_RESOURCE_API_KEY"),
    )

    # Agent configuration list: (env_var, agent_name, deps_type)
    configs = [
        ("AZURE_MODEL_NAME_MA", "meta_agent", None),
        ("AZURE_MODEL_NAME_UA", "Updater", None),
        ("AZURE_MODEL_NAME_RA", "Reviewer", review_agent_deps),
        ("AZURE_MODEL_NAME_WA", "Writer", None),
    ]

    # Hold instantiated models/agents
    models = {}
    agents = {}

    for env_var, name, deps in configs:
        # Initialize model
        model = OpenAIModel(
            model_name=os.getenv(env_var),
            openai_client=azure,
        )
        
        # Instantiate agent
        agents[name] = Agent(
            model=model,
            name=name,
            retries=3,
            deps_type=deps,
        )

        models[name] = model

    return Clients(
        supabase_client=supabase,
        azure_client=azure,

        model_update=models["Updater"],
        update_agent=agents["Updater"],
        model_meta=models["meta_agent"],
        meta_agent=agents["meta_agent"],
        model_reviewer=models["Reviewer"],
        reviewer_agent=agents["Reviewer"],
        model_writer=models["Writer"],
        writer_agent=agents["Writer"],
    )





