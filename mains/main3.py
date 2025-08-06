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
OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MAX_AGENT_LOOPS = 7  # Max steps the agent can take

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hybrid Data Analyst Agent",
    description="A robust agent that plans, installs dependencies, and then uses a toolbox of functions to answer data analysis questions.",
)

# --- AGENT PROMPTS ---

# Phase 1 Prompt: Planner
PLANNER_SYSTEM_PROMPT = """
You are a lead data analyst responsible for planning. Based on the user's request, identify all necessary Python libraries that need to be installed via pip for the entire task.
Your output MUST be a single, valid JSON object with one key: "libraries".
- "libraries": A list of strings for pip to install (e.g., "pandas", "duckdb"). Do NOT include standard libraries that are built into Python (e.g., json, re, base64, os).
"""

# Phase 2 Prompt: Main Agent Brain
AGENT_SYSTEM_PROMPT = """
You are a world-class autonomous data analyst agent. Your goal is to solve the user's request by calling a sequence of tools.
You must operate in a loop of Thought -> Action. At each step, you will first state your reasoning, then choose a tool to use.

**TOOLS:**
You have access to the following tools. You must respond with a JSON object that specifies the tool and its arguments.

1. `run_sql_query(query: str)`: Executes a SQL query using DuckDB against the S3 dataset. Use this for initial data loading, filtering, and aggregations. The table name is `read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')`.

2. `scrape_website(url: str)`: Scrapes the text content of a URL.

3. `execute_python_code(code: str)`: Executes arbitrary Python code for complex tasks like data cleaning, transformations, regressions, and plotting. The code MUST be self-contained and import any necessary libraries. The last line of the code must be an expression or a variable that holds the result to be returned.

4. `finish(answer_json: str)`: When you have the final answer that satisfies the user's request, call this tool with the complete JSON object containing all the answers.

**RESPONSE FORMAT:**
You MUST respond with a single JSON object containing "thought" and "action". The "action" object must contain "tool_name" and "arguments".

Example:
{
  "thought": "I need to find out which high court disposed the most cases. I will use a SQL query to group by court and count the cases between 2019 and 2022.",
  "action": {
    "tool_name": "run_sql_query",
    "arguments": {
      "query": "SELECT court, COUNT(*) as case_count FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1') WHERE year BETWEEN 2019 AND 2022 GROUP BY court ORDER BY case_count DESC LIMIT 1;"
    }
  }
}
"""

# --- Helper Functions ---


def install_package(package_name: str):
    """Installs a package using pip. Raises an exception on failure."""
    try:
        logger.info(f"Installing package: {package_name}")
        subprocess.run(
            ["python", "-m", "pip", "install", package_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        logger.info(f"Successfully installed {package_name}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e.stderr}")
        raise  # Re-raise the exception to be caught by the main handler


def run_sql_query(query: str) -> str:
    """Executes a SQL query using DuckDB and returns the result as a JSON string."""
    try:
        import duckdb
        import pandas as pd

        logger.info(f"Executing SQL Query: {query[:200]}...")
        con = duckdb.connect(database=":memory:")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")
        result_df = con.execute(query).fetchdf()
        return result_df.to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": f"Failed to execute SQL query: {e}"})


def scrape_website(url: str) -> str:
    """Scrapes the text content of a given URL."""
    try:
        logger.info(f"Scraping URL: {url}")
        response = httpx.get(url, follow_redirects=True, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        return soup.body.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        return json.dumps({"error": f"Failed to scrape URL: {e}"})


def execute_python_code(code: str) -> str:
    """Executes a string of Python code and returns its output."""
    logger.info(f"Executing Python Code:\n---\n{code[:500]}...\n---")
    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return json.dumps(
                {
                    "error": "Python code execution failed",
                    "details": result.stderr.strip(),
                }
            )
        return result.stdout.strip()
    except Exception as e:
        return json.dumps(
            {"error": f"An unexpected error occurred during execution: {e}"}
        )


# --- Core Logic ---


def get_tool_call_from_response(response_text: str) -> dict | None:
    json_str = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_str:
        try:
            return json.loads(json_str.group(0))
        except json.JSONDecodeError:
            return None
    return None


TOOLS = {
    "run_sql_query": run_sql_query,
    "scrape_website": scrape_website,
    "execute_python_code": execute_python_code,
}


async def call_llm(messages: list) -> str:
    aiproxy_token = os.environ.get("OPENAI_API_KEY")
    if not aiproxy_token:
        raise HTTPException(
            status_code=500, detail="OPENAI_API_KEY environment variable is not set."
        )

    headers = {
        "Authorization": f"Bearer {aiproxy_token}",
        "Content-Type": "application/json",
    }
    payload = {"model": "openai/gpt-4o-mini", "messages": messages, "temperature": 0.0}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OPENAI_BASE_URL, json=payload, headers=headers, timeout=120
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
    task_description = (await form_data.get("file").read()).decode("utf-8")
    logger.info(f"Received task: {task_description}")

    # --- PHASE 1: PLANNING & INSTALLATION ---
    logger.info("--- Entering Planning & Installation Phase ---")
    try:
        plan_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ]
        plan_response_str = await call_llm(plan_messages)
        plan_json = get_tool_call_from_response(plan_response_str)
        libraries = plan_json.get("libraries", [])

        logger.info(f"Planner identified required libraries: {libraries}")

        core_libs = {"duckdb", "pandas", "scikit-learn", "matplotlib", "seaborn"}
        required_libs = set(libraries) | core_libs

        for lib in required_libs:
            install_package(lib)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed during planning/installation phase: {e}"
        )

    # --- PHASE 2: TOOL-USING EXECUTION LOOP ---
    logger.info("--- Entering Execution Phase ---")
    conversation_history = [{"role": "user", "content": task_description}]
    for i in range(MAX_AGENT_LOOPS):
        logger.info(f"--- Agent Loop {i + 1}/{MAX_AGENT_LOOPS} ---")

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ] + conversation_history
        llm_response_str = await call_llm(messages)
        tool_call = get_tool_call_from_response(llm_response_str)

        if not tool_call or "action" not in tool_call:
            raise HTTPException(
                status_code=500,
                detail=f"Agent did not produce a valid action on loop {i+1}.",
            )

        action = tool_call.get("action", {})
        tool_name = action.get("tool_name")
        arguments = action.get("arguments", {})
        thought = tool_call.get("thought", "")

        logger.info(f"Thought: {thought}")
        logger.info(f"Action: Calling tool '{tool_name}' with args {arguments}")

        if tool_name == "finish":
            final_answer = arguments.get("answer_json")
            try:
                return json.loads(final_answer)
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Agent returned malformed final JSON.",
                        "output": final_answer,
                    },
                )

        if tool_name in TOOLS:
            tool_function = TOOLS[tool_name]
            try:
                observation = tool_function(**arguments)
            except TypeError as e:
                observation = json.dumps(
                    {
                        "error": f"Invalid arguments for tool {tool_name}",
                        "details": f"{e}",
                    }
                )

            conversation_history.append(
                {"role": "assistant", "content": json.dumps(tool_call, indent=2)}
            )
            # THIS IS THE FIX: Truncate the observation to keep the context small
            conversation_history.append(
                {"role": "tool", "content": observation[:3000], "name": tool_name}
            )
        else:
            observation = json.dumps({"error": f"Unknown tool specified: {tool_name}"})
            conversation_history.append(
                {"role": "tool", "content": observation, "name": "error"}
            )

    raise HTTPException(
        status_code=500, detail="Agent exceeded maximum number of loops."
    )
