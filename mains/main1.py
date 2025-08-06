import os
import re
import subprocess
import json
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
import logging

# --- Configuration ---
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
MAX_RETRIES = 3

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent",
    description="An agent that uses an LLM to write and execute code to answer data analysis questions.",
)

# --- System & Fixer Prompts ---

# STEP 1: The Planner Prompt
PLANNER_SYSTEM_PROMPT = """
You are a lead data analyst responsible for planning tasks. Based on the user's request, create a clear, step-by-step plan and identify all necessary Python libraries to be installed via pip.
Your output MUST be a single, valid JSON object with two keys: "plan" and "libraries".
- "plan": A list of strings, where each string is a logical step in the analysis.
- "libraries": A list of strings for pip to install (e.g., "pandas", "duckdb"). Do NOT include standard libraries that are built into Python (e.g., json, re, base64, os).
"""

# STEP 2: The Coder Prompt
CODER_SYSTEM_PROMPT = """
You are an expert Python coder. Your goal is to write a complete, self-contained Python script to execute the provided plan.

**CORE DIRECTIVES:**
1.  **Follow the Plan**: Implement the step-by-step plan exactly.
2.  **Be Robust**: Write clean, efficient, and well-documented code. Always include data validation and error handling.
3.  **Self-Contained Script**: Your script must import all necessary libraries at the beginning.

**CODE GENERATION RULES:**
* You MUST write a single, runnable Python script.
* Enclose the final Python code within a single markdown block: ```python\n...code...\n```.

**CRITICAL OUTPUT RULE:**
* Your script's final action MUST be a single `print()` statement to standard output containing the complete answer, formatted as a single JSON string using `json.dumps()`.
* **Infer the JSON structure from the user's original request.**
* **NEVER give up and return a JSON object containing an 'error' key.** If you encounter an error you cannot fix, it is better to let the script fail with an exception than to return a JSON error message. The control loop will handle the error.
"""

FIXER_PROMPT_TEMPLATE = """
The Python code you previously wrote, based on the user's request, failed to execute. Review the code, the original request, and the error, then provide a corrected version.

**Original User Request:**
{task_description}

**Original Code:**
```python
{previous_code}
Error Message:
{error_message}

Your Task:
Fix the bug in the original code to successfully complete the user's request. Do not give up. For IOException on a file path, double-check the path syntax from the user's prompt, especially wildcards.

Provide the complete, corrected, and robust Python script below.
"""


# --- Helper Functions ---
def extract_json_object(response_text: str) -> str | None:
    match = re.search(r"{.*}", response_text, re.DOTALL)
    return match.group(0) if match else None


def extract_python_code(response_text: str) -> str | None:
    match = re.search(r"(?:python)?\n(.*?)\n", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


def install_package(package_name: str) -> tuple[bool, str | None]:
    try:
        logger.info(f"Checking/installing package: {package_name}")
        subprocess.run(
            ["python", "-m", "pip", "install", package_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        logger.info(f"Successfully installed {package_name}.")
        return True, None
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.strip()
        logger.error(f"Failed to install {package_name}: {error_message}")
        return False, error_message
    except Exception as e:
        logger.error(f"An exception occurred during package installation: {e}")
        return False, str(e)


def get_page_structure(url: str) -> str | None:
    try:
        response = httpx.get(url, follow_redirects=True, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        body_text = soup.body.get_text(separator=" ", strip=True)[:1000]
        return f"The webpage at {url} has the following text snippet for context: \n---\n{body_text}\n---\n"
    except Exception as e:
        logger.error(f"Failed to scrape URL {url}: {e}")
        return f"Could not scrape the URL: {url}. Error: {e}"


def run_code(script: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["python", "-c", script], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return result.stdout.strip(), None
    except Exception as e:
        return None, f"An unexpected error occurred during execution: {e}"


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    aiproxy_token = os.environ.get("AIPROXY_TOKEN")

    if not aiproxy_token:
        raise HTTPException(
            status_code=500, detail="AIPROXY_TOKEN environment variable is not set."
        )

    headers = {
        "Authorization": f"Bearer {aiproxy_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                AIPROXY_URL, json=payload, headers=headers, timeout=90
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, KeyError, IndexError) as e:
            logger.error(f"LLM API request failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {e}")


# --- API Endpoint ---
@app.post("/api/")
async def data_analyst_agent(request: Request):
    form_data = await request.form()
    uploaded_file = form_data.get("file")

    if not uploaded_file:
        raise HTTPException(
            status_code=400, detail="No task provided in 'file' form field."
        )

    task_description = (await uploaded_file.read()).decode("utf-8")
    logger.info(f"Received task: {task_description}")

    # --- PHASE 1: PLANNING ---
    logger.info("--- Entering Planning Phase ---")
    try:
        plan_response_str = await call_llm(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=task_description,
        )
        json_str = extract_json_object(plan_response_str)
        if not json_str:
            raise ValueError("LLM did not return a JSON object for the plan.")

        plan_response = json.loads(json_str)
        plan = plan_response.get("plan", [])
        libraries = plan_response.get("libraries", [])
        if not plan:
            raise ValueError("Planner returned JSON but without a 'plan' key.")
        logger.info(f"Plan received: {plan}")
        logger.info(f"Required libraries: {libraries}")
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to get a valid plan from the LLM: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent could not devise a plan for this task. Reason: {e}",
        )

    # --- PHASE 2: DEPENDENCY INSTALLATION ---
    logger.info("--- Entering Dependency Installation Phase ---")
    installation_notes = []
    if libraries:
        for lib in libraries:
            success, error_msg = install_package(lib)
            if not success:
                if error_msg and "No matching distribution found" in error_msg:
                    note = f"- The package '{lib}' was not found on PyPI. It is likely a Python built-in module and can be imported directly."
                    logger.warning(note)
                    installation_notes.append(note)
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to install required dependency '{lib}': {error_msg}",
                    )

    # --- PHASE 3: CODING & EXECUTION ---
    logger.info("--- Entering Coding & Execution Phase ---")

    url_match = re.search(r"https?://[^\s)]+", task_description)
    page_context = ""
    if url_match:
        url = url_match.group(0)
        page_context = get_page_structure(url)

    coder_prompt = f"**Original User Request:**\n{task_description}\n\n**Execution Plan:**\n{json.dumps(plan, indent=2)}"
    if page_context:
        coder_prompt += f"\n\n**Webpage Context:**\n{page_context}"
    if installation_notes:
        coder_prompt += f"\n\n**Dependency Installation Notes:**\n" + "\n".join(
            installation_notes
        )

    generated_code = None
    for attempt in range(MAX_RETRIES):
        logger.info(f"Agent attempt {attempt + 1}/{MAX_RETRIES}...")

        if not generated_code:
            llm_response = await call_llm(CODER_SYSTEM_PROMPT, coder_prompt)
            generated_code = extract_python_code(llm_response)

            if not generated_code:
                logger.warning("LLM Coder did not return valid code. Retrying.")
                coder_prompt = "You did not return a Python script. Please follow the plan and provide the full script."
                continue

        logger.info(f"Code to execute:\n---\n{generated_code[:500]}...\n---")
        stdout, stderr = run_code(generated_code)

        if stderr:
            logger.error(f"Code execution failed with error:\n{stderr}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Agent failed after multiple attempts.",
                        "last_error": stderr,
                    },
                )

            coder_prompt = FIXER_PROMPT_TEMPLATE.format(
                previous_code=generated_code,
                error_message=stderr,
                task_description=task_description,
            )
            generated_code = None
            continue

        logger.info(f"Code executed successfully. Output:\n{stdout}")
        try:
            final_result = json.loads(stdout)
            if isinstance(final_result, dict) and "error" in final_result:
                error_from_agent = final_result["error"]
                logger.error(f"Agent returned a JSON error object: {error_from_agent}")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(status_code=500, detail=final_result)

                coder_prompt = f"""Your previous attempt failed because you returned an error message as the result. Do not do this. You must fix the root cause of the error.
The original user request was: {task_description}
The error was: {error_from_agent}
The code that produced the error was:

Python

{generated_code}
Please try again and fix the underlying issue."""
                generated_code = None
                continue

            return final_result

        except json.JSONDecodeError:
            logger.error(f"CRITICAL: Agent script produced malformed JSON: {stdout}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Agent script produced non-JSON output.",
                        "output": stdout,
                    },
                )

            coder_prompt = f"""Your script executed without crashing but produced output that was not valid JSON. This is unacceptable. You MUST solve the original user's request and your final output MUST be a valid JSON string. Do not just output a sample JSON.
The invalid output was: "{stdout}"

The code that produced this was:

Python

{generated_code}
The original user request was:
{task_description}

Please rewrite the script to correctly solve the original request and produce valid JSON."""
            generated_code = None
            continue

    raise HTTPException(
        status_code=500, detail="Agent failed to produce a valid result."
    )
