import os
import re
import json
from fastapi import FastAPI, HTTPException, Request
import httpx
import logging
from pathlib import Path

# --- Configuration ---
OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent - Code Generator",
    description="A two-step agent that first creates a technical plan and then generates Python code from it.",
)

# --- Helper Functions ---


def load_prompt(file_path: str) -> str:
    """Loads a prompt string from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found at path: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt file {file_path}: {e}")
        raise


# --- Load Prompts from Files on Startup ---
# This is more efficient as it's done only once when the app starts.
try:
    PROMPTS_DIR = Path(__file__).parent / "prompts"
    METADATA_EXTRACTOR_PROMPT = load_prompt(PROMPTS_DIR / "extractor_prompt.txt")
    CODER_SYSTEM_PROMPT = load_prompt(PROMPTS_DIR / "coder_prompt.txt")
except Exception:
    logger.critical(
        "Could not load necessary prompt files. The application cannot start."
    )
    # In a real application, you might want a more graceful shutdown.
    exit(1)


def extract_python_code(response_text: str) -> str | None:
    """Extracts Python code from a markdown block."""
    match = re.search(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Calls the specified LLM API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY environment variable is not set."
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OPENAI_BASE_URL, json=payload, headers=headers, timeout=90
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, KeyError, IndexError) as e:
            logger.error(f"LLM API request failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {e}")


# --- API Endpoint ---
@app.post("/generate_code")
async def generate_code_endpoint(request: Request):
    """
    Receives a task, creates a technical plan, and then generates the Python code to solve it.
    """
    form_data = await request.form()
    file_upload = form_data.get("file")
    if not file_upload:
        raise HTTPException(
            status_code=400, detail="No file provided in 'file' form field."
        )

    task_description = (await file_upload.read()).decode("utf-8")

    # --- STAGE 1: TECHNICAL PLAN GENERATION ---
    logger.info("--- Stage 1: Generating Technical Plan ---")
    try:
        technical_plan = await call_llm(
            system_prompt=METADATA_EXTRACTOR_PROMPT,
            user_prompt=task_description,
        )
        if not technical_plan or not technical_plan.strip():
            raise ValueError("Extractor LLM did not return a valid plan.")

        logger.info(f"Successfully generated technical plan:\n{technical_plan}")
    except ValueError as e:
        logger.error(f"Failed to get a valid plan from the Extractor LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Stage 1 Failure: {e}")

    # --- STAGE 2: CODE GENERATION ---
    logger.info("--- Stage 2: Generating Code from Plan ---")
    try:
        coder_prompt = technical_plan

        code_response_str = await call_llm(
            system_prompt=CODER_SYSTEM_PROMPT,
            user_prompt=coder_prompt,
        )

        generated_code = extract_python_code(code_response_str)
        if not generated_code:
            raise ValueError("Coder LLM did not return a valid Python script.")

        logger.info(f"Successfully generated code:\n{generated_code[:500]}...")
        return {"generated_code": generated_code}

    except ValueError as e:
        logger.error(f"Failed to generate code from the Coder LLM: {e}")
        raise HTTPException(status_code=500, detail=f"Stage 2 Failure: {e}")
