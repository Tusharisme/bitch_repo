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
SYSTEM_PROMPT = """
You are a world-class autonomous data analyst agent. Your goal is to write a complete, self-contained Python script to answer the user's request.

**CORE DIRECTIVES:**

1.  **Deconstruct the Task**: First, carefully analyze the user's request. Identify the core questions, the data source (e.g., URL for scraping, S3 path for Parquet files), the required tools (e.g., pandas, DuckDB), and the specific output format (e.g., JSON array, JSON object).

2.  **Source and Load Data**:
    * **Web Scraping**: If a URL is provided, use libraries like `requests` and `BeautifulSoup`. Be resilient to changes in HTML structure.
    * **Remote Files (Parquet/CSV)**: If an S3 path and a tool like DuckDB are mentioned, use them. Construct the correct query to read the data.
    * **Local Files**: If the user provides data directly, use `pandas` or the appropriate library to load it.

3.  **Inspect and Clean Data (MANDATORY)**: This is the most critical step. Never trust raw data.
    * **Always** inspect the data's structure (`.info()`, `.head()`).
    * **Create robust cleaning functions** as needed. For example, if you see monetary values like "$1.5 billion" or "₹5 Lakhs", write a function to convert them to a consistent numerical format (e.g., float). Do the same for dates or other non-standard types.
    * Handle missing values (`NaN`) appropriately.

4.  **Analyze Data**: Based on the user's questions, perform the required analysis. This could include:
    * Filtering and querying data (`pandas` queries, SQL `WHERE` clauses).
    * Aggregations (`.groupby()`, `COUNT()`, `SUM()`).
    * Statistical analysis (correlation, regression).
    * Time-series analysis.

5.  **Visualize Data**: If asked to create a plot:
    * Use `matplotlib` or `seaborn`.
    * Ensure the plot is clear, with labeled axes and a title.
    * The plot MUST be saved to a `io.BytesIO` buffer and encoded into a `data:image/png;base64,...` string.

**CODE GENERATION RULES:**
* You MUST write a single, self-contained, and runnable Python script.
* Import all necessary libraries at the beginning of the script.
* Enclose the final Python code within a single markdown block: ```python\n...code...\n```.

**CRITICAL OUTPUT RULE:**
* Your script's final action MUST be a single `print()` statement to standard output.
* This `print()` MUST contain the complete answer, formatted as a single JSON string.
* **Infer the JSON structure from the user's request.** If they ask for a list of answers, produce a JSON array. If they provide a JSON object structure with questions as keys, you MUST match that structure.
* Use `json.dumps()` to ensure correct formatting.
* NEVER print anything else to stdout - only the final JSON result.
* Always test your data cleaning and conversion logic with try-except blocks.
* Before the final print(), validate that your result makes sense and contains the expected data.
"""

FIXER_PROMPT_TEMPLATE = """
The Python code you previously wrote failed to execute. Review the code and the error, and provide a corrected version.

**Original Code:**
```python
{previous_code}
```

**Error Message:**
{error_message}

**DEBUGGING CHECKLIST:**

1. **Read the Error Carefully**: What line and type of error occurred (IndexError, KeyError, ValueError)?

2. **Check Data Access**:
   - If IndexError: Did you check if the list or DataFrame was empty before accessing an element by index (e.g., if len(my_list) > 0:)?
   - If KeyError: Did you verify the column name or dictionary key exists before trying to use it?

3. **Check Data Cleaning & Types**:
   - If ValueError: Did your data cleaning functions handle all edge cases (e.g., different currency formats, unexpected text in a number column)? Did you try-except the type conversion?

4. **Check Scraping Logic**: Did the HTML structure change? Are your CSS selectors still valid? Add defensive checks for empty results.

5. **Review Logic**: Does your analysis code logically lead to the result?

Provide the complete, corrected, and robust Python script below, ensuring it adheres to all original directives, especially the CRITICAL OUTPUT RULE.
"""


# --- Helper Functions ---
def extract_python_code(response_text: str) -> str | None:
    """Extracts Python code from a markdown block, optionally matching 'python'."""
    # Try multiple patterns to be more robust
    patterns = [
        r"```python\n(.*?)\n```",
        r"```\n(.*?)\n```",
        r"```python(.*?)```",
        r"```(.*?)```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:  # Make sure we got actual content
                return code

    return None


def get_page_structure(url: str) -> str | None:
    """Fetches a webpage and returns its simplified tag structure."""
    try:
        response = httpx.get(url, follow_redirects=True, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        body_text = soup.body.get_text(separator=" ", strip=True)[:500]
        return f"The webpage at {url} has the following text snippet for context: \n---\n{body_text}\n---\n"
    except Exception as e:
        logger.error(f"Failed to scrape URL {url}: {e}")
        return f"Could not scrape the URL: {url}. Error: {e}"


def run_code(script: str) -> tuple[str | None, str | None]:
    """Executes a Python script as a string in a separate process."""
    try:
        result = subprocess.run(
            ["python", "-c", script], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return result.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "The script execution timed out after 120 seconds."
    except Exception as e:
        return None, f"An unexpected error occurred during execution: {e}"


async def call_llm(user_prompt: str) -> str:
    """Calls the AI Proxy LLM."""
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
            {"role": "system", "content": SYSTEM_PROMPT},
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
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API request failed: {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"LLM API error: {e.response.text}",
            )
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to communicate with the LLM: {str(e)}"
            )


# --- API Endpoint ---
@app.post("/api/")
async def data_analyst_agent(request: Request):
    """The main endpoint that takes a data analysis task and returns the result."""
    form_data = await request.form()
    uploaded_file = form_data.get("file")

    if not uploaded_file:
        raise HTTPException(
            status_code=400, detail="No task provided in the 'file' form field."
        )

    file_content = await uploaded_file.read()
    task_description = file_content.decode("utf-8")

    logger.info(f"Received task: {task_description}")

    url_match = re.search(r"https?://\S+", task_description)
    page_context = ""
    if url_match:
        url = url_match.group(0).rstrip(')"')
        logger.info(f"URL detected: {url}. Fetching page structure.")
        page_context = get_page_structure(url)

    current_prompt = f"**Full User Request:**\n{task_description}\n\n"
    if page_context:
        current_prompt += f"**Webpage Context:**\n{page_context}"

    generated_code = None

    for attempt in range(MAX_RETRIES):
        logger.info(f"Agent attempt {attempt + 1}/{MAX_RETRIES}...")

        llm_response = await call_llm(current_prompt)
        generated_code = extract_python_code(llm_response)

        if not generated_code:
            logger.warning("LLM did not return valid code. Retrying.")
            current_prompt = f"You did not return a Python script in the required format. Please read the user request carefully and provide the complete script inside a single ```python block.\n\n**Full User Request:**\n{task_description}"
            continue

        logger.info(
            f"Code generated. Executing now:\n---\n{generated_code[:500]}...\n---"
        )
        stdout, stderr = run_code(generated_code)

        if stderr:
            logger.error(f"Code execution failed with error:\n{stderr}")
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Agent failed to write working code after multiple attempts.",
                        "last_error": stderr,
                    },
                )

            current_prompt = FIXER_PROMPT_TEMPLATE.format(
                previous_code=generated_code, error_message=stderr
            )
        else:
            logger.info(f"Code executed successfully. Output:\n{stdout}")

            # Check if output is empty
            if not stdout or stdout.strip() == "":
                logger.error("CRITICAL: Agent script produced empty output")
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "Agent script produced empty output after multiple attempts.",
                            "last_code": generated_code,
                        },
                    )
                current_prompt = f"""The script executed without errors but produced no output. The script must end with a print() statement that outputs a JSON string.

**Original Code:**
```python
{generated_code}
```

Fix the script to ensure it prints the final JSON result. The task was: {task_description}"""
                continue

            try:
                final_result = json.loads(stdout)
                return final_result
            except json.JSONDecodeError as e:
                logger.error(
                    f"CRITICAL: Agent script produced malformed JSON: {stdout}"
                )
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "Agent script produced non-JSON output.",
                            "output": stdout,
                        },
                    )
                current_prompt = f"""The script executed but produced invalid JSON output: "{stdout}"

**Original Code:**
```python
{generated_code}
```

**JSON Error:** {str(e)}

Fix the script to ensure it outputs valid JSON. The final print() statement must output a proper JSON string using json.dumps(). The task was: {task_description}"""

    raise HTTPException(
        status_code=500,
        detail="Agent failed to produce a valid result within the retry limit.",
    )


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Data Analyst Agent is running. POST your a task to /api/"}
