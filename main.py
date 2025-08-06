import os
import re
import json
import subprocess
from fastapi import FastAPI, HTTPException, Request
import httpx
import logging
from pathlib import Path

# --- Configuration ---
OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent - Code Generator & Executor",
    description="A two-step agent that creates, corrects, and executes code to fulfill user data tasks.",
)


# --- Load Prompts ---
def load_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


try:
    PROMPTS_DIR = Path(__file__).parent / "prompts"
    METADATA_EXTRACTOR_PROMPT = load_prompt(PROMPTS_DIR / "extractor_prompt.txt")
    CODER_SYSTEM_PROMPT = load_prompt(PROMPTS_DIR / "coder_prompt.txt")
    FIX_CODE_PROMPT_TEMPLATE = load_prompt(PROMPTS_DIR / "fix_code_prompt.txt")
except Exception as e:
    logger.critical(f"Prompt loading failed: {e}")
    exit(1)


# --- Utility Functions ---
def extract_python_code(response_text: str) -> str:
    # This regex is intentionally simple to capture only the code
    # It avoids capturing potential explanations after the closing backticks.
    match = re.search(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_dependencies(code: str) -> list[str]:
    # A simple regex to find a list named 'dependencies'
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", code)
    if match:
        # This part cleans up the matched string into a clean list of package names
        return [
            pkg.strip().strip('"').strip("'")
            for pkg in match.group(1).split(",")
            if pkg.strip()
        ]
    return []


def install_dependencies(deps: list[str]):
    if not deps:
        logger.info("No dependencies found to install.")
        return
    for dep in deps:
        logger.info(f"Installing {dep}...")
        # Using check=True will raise an exception if the command fails
        subprocess.run(["uv", "pip", "install", dep], check=True, capture_output=True)


def run_code_with_uv(code: str) -> dict:
    try:
        # uv run executes a command within a temporary virtual environment
        result = subprocess.run(
            ["uv", "run", "-"],
            input=code.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.decode("utf-8"),
            "stderr": result.stderr.decode("utf-8"),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Execution timed out."}
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"An unexpected error occurred during execution: {e}",
        }


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

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
            logger.error(f"LLM call failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {e}")


async def fix_code_with_llm(code: str, error: str) -> str:
    fix_prompt = FIX_CODE_PROMPT_TEMPLATE.replace("{{code}}", code).replace(
        "{{error}}", error
    )
    response = await call_llm(CODER_SYSTEM_PROMPT, fix_prompt)

    # ADDED FOR LOGGING
    logger.info(f"--- Raw Fixer Response ---\n{response}")

    return extract_python_code(response)


# --- API Endpoint ---
@app.post("/generate_and_run")
async def generate_and_run(request: Request):
    form_data = await request.form()
    file_upload = form_data.get("file")
    if not file_upload:
        raise HTTPException(status_code=400, detail="No file uploaded under 'file'.")

    user_task = (await file_upload.read()).decode("utf-8")

    logger.info("--- Generating Technical Plan ---")
    plan = await call_llm(METADATA_EXTRACTOR_PROMPT, user_task)

    # ADDED FOR LOGGING
    logger.info(f"--- Plan Generated ---\n{plan}")

    logger.info("--- Generating Code ---")
    code_block = await call_llm(CODER_SYSTEM_PROMPT, plan)

    # ADDED FOR LOGGING
    logger.info(f"--- Raw Code Block Generated ---\n{code_block}")

    code = extract_python_code(code_block)

    if not code:
        raise HTTPException(
            status_code=500, detail="Failed to extract code from LLM response."
        )

    attempt = 0
    while attempt < MAX_RETRIES:
        logger.info(f"--- Attempt {attempt + 1}: Installing Dependencies ---")
        try:
            deps = extract_dependencies(code)
            install_dependencies(deps)
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            # We can still try to run the code, it might not need installation

        logger.info("--- Running Code ---")
        result = run_code_with_uv(code)

        if result["success"]:
            logger.info(f"--- Code Execution Successful ---\n{result['stdout']}")
            return {"output": result["stdout"]}

        logger.warning(f"--- Code Execution Failed ---\n{result['stderr']}")

        if attempt >= MAX_RETRIES - 1:
            # Break the loop if this is the last attempt
            break

        logger.info("--- Asking LLM to Fix Code ---")
        new_code = await fix_code_with_llm(code, result["stderr"])
        if new_code:
            code = new_code
        else:
            logger.error(
                "LLM failed to provide a valid code fix. Retrying with original code."
            )

        attempt += 1

    raise HTTPException(
        status_code=500, detail="Code failed after multiple LLM fix attempts."
    )
