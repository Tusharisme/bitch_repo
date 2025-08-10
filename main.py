# import os
# import re
# import json
# import subprocess
# import ast
# from fastapi import FastAPI, HTTPException, Request
# import httpx
# import logging
# from pathlib import Path

# # --- Configuration ---
# OPENAI_BASE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
# MAX_RETRIES = 3

# # --- Logging Setup ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="Data Analyst Agent - Code Generator & Executor",
#     description="A two-step agent that creates, corrects, and executes code to fulfill user data tasks.",
# )


# # --- Load Prompts ---
# def load_prompt(file_path: str) -> str:
#     with open(file_path, "r", encoding="utf-8") as f:
#         return f.read()


# try:
#     PROMPTS_DIR = Path(__file__).parent / "prompts"
#     METADATA_EXTRACTOR_PROMPT = load_prompt(PROMPTS_DIR / "extractor_prompt.txt")
#     CODER_SYSTEM_PROMPT = load_prompt(PROMPTS_DIR / "coder_prompt.txt")
#     FIX_CODE_PROMPT_TEMPLATE = load_prompt(PROMPTS_DIR / "fix_code_prompt.txt")
# except Exception as e:
#     logger.critical(f"Prompt loading failed: {e}")
#     exit(1)


# # --- Utility Functions ---
# def extract_python_code(response_text: str) -> str:
#     match = re.search(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
#     return match.group(1).strip() if match else None


# def extract_dependencies(code: str) -> list[str]:
#     match = re.search(r"dependencies\s*=\s*\[(.*?)\]", code)
#     if match:
#         return [
#             pkg.strip().strip('"').strip("'")
#             for pkg in match.group(1).split(",")
#             if pkg.strip()
#         ]
#     return []


# def extract_imports_from_code(code: str) -> list[str]:
#     """Parse code to find imported modules."""
#     try:
#         tree = ast.parse(code)
#     except SyntaxError:
#         return []
#     imports = set()
#     for node in ast.walk(tree):
#         if isinstance(node, ast.Import):
#             for alias in node.names:
#                 imports.add(alias.name.split(".")[0])
#         elif isinstance(node, ast.ImportFrom):
#             if node.module:
#                 imports.add(node.module.split(".")[0])
#     return list(imports)


# def install_dependencies(deps: list[str]):
#     if not deps:
#         logger.info("No dependencies found to install.")
#         return
#     for dep in deps:
#         logger.info(f"Installing {dep}...")
#         try:
#             result = subprocess.run(
#                 ["uv", "pip", "install", dep],
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#             )
#             if result.returncode != 0:
#                 logger.warning(f"'uv' failed for {dep}. Trying pip...")
#                 pip_result = subprocess.run(
#                     ["pip", "install", dep],
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                 )
#                 if pip_result.returncode != 0:
#                     logger.error(
#                         f"Both uv and pip failed to install {dep}.\n"
#                         f"uv stderr: {result.stderr.decode()}\n"
#                         f"pip stderr: {pip_result.stderr.decode()}"
#                     )
#         except Exception as e:
#             logger.error(f"Unexpected error installing {dep}: {e}")


# def run_code_with_uv(code: str) -> dict:
#     try:
#         result = subprocess.run(
#             ["uv", "run", "-"],
#             input=code.encode("utf-8"),
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             timeout=60,
#         )
#         return {
#             "success": result.returncode == 0,
#             "stdout": result.stdout.decode("utf-8"),
#             "stderr": result.stderr.decode("utf-8"),
#         }
#     except subprocess.TimeoutExpired:
#         return {"success": False, "stdout": "", "stderr": "Execution timed out."}
#     except Exception as e:
#         return {
#             "success": False,
#             "stdout": "",
#             "stderr": f"An unexpected error occurred during execution: {e}",
#         }


# async def call_llm(system_prompt: str, user_prompt: str) -> str:
#     api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM3NTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.gd84YB3ekhIQzU4E4Tgogaeya9idpV6hMTi6uPja7j0"
#     if not api_key:
#         raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

#     headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#     payload = {
#         "model": "openai/gpt-4o-mini",
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": 0.0,
#     }

#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.post(
#                 OPENAI_BASE_URL, json=payload, headers=headers, timeout=90
#             )
#             response.raise_for_status()
#             return response.json()["choices"][0]["message"]["content"]
#         except (httpx.HTTPStatusError, KeyError, IndexError) as e:
#             logger.error(f"LLM call failed: {e}")
#             raise HTTPException(status_code=500, detail=f"LLM API error: {e}")


# async def fix_code_with_llm(code: str, error: str) -> str:
#     fix_prompt = FIX_CODE_PROMPT_TEMPLATE.replace("{{code}}", code).replace(
#         "{{error}}", error
#     )
#     response = await call_llm(CODER_SYSTEM_PROMPT, fix_prompt)
#     logger.info(f"--- Raw Fixer Response ---\n{response}")
#     return extract_python_code(response)


# # --- API Endpoint ---
# @app.post("/generate_and_run")
# async def generate_and_run(request: Request):
#     form_data = await request.form()
#     file_upload = form_data.get("file")
#     if not file_upload:
#         raise HTTPException(status_code=400, detail="No file uploaded under 'file'.")

#     user_task = (await file_upload.read()).decode("utf-8")

#     logger.info("--- Generating Technical Plan ---")
#     plan = await call_llm(METADATA_EXTRACTOR_PROMPT, user_task)
#     logger.info(f"--- Plan Generated ---\n{plan}")

#     logger.info("--- Generating Code ---")
#     code_block = await call_llm(CODER_SYSTEM_PROMPT, plan)
#     logger.info(f"--- Raw Code Block Generated ---\n{code_block}")

#     code = extract_python_code(code_block)
#     if not code:
#         raise HTTPException(
#             status_code=500, detail="Failed to extract code from LLM response."
#         )

#     attempt = 0
#     while attempt < MAX_RETRIES:
#         logger.info(f"--- Attempt {attempt + 1}: Installing Dependencies ---")
#         try:
#             deps = set(extract_dependencies(code)) | set(
#                 extract_imports_from_code(code)
#             )
#             install_dependencies(list(deps))
#         except Exception as e:
#             logger.error(f"Dependency installation failed: {e}")

#         logger.info("--- Running Code ---")
#         result = run_code_with_uv(code)

#         # If execution succeeded, return output
#         if result["success"]:
#             logger.info(f"--- Code Execution Successful ---\n{result['stdout']}")
#             return {"output": result["stdout"]}

#         logger.warning(f"--- Code Execution Failed ---\n{result['stderr']}")

#         # Check if it's a missing module and install it directly
#         missing_module_match = re.search(
#             r"ModuleNotFoundError: No module named '([^']+)'", result["stderr"]
#         )
#         if missing_module_match:
#             missing_pkg = missing_module_match.group(1)
#             logger.info(f"Detected missing package: {missing_pkg}. Installing...")
#             try:
#                 install_dependencies([missing_pkg])
#                 continue  # Retry same code without LLM fix
#             except Exception as e:
#                 logger.error(f"Failed to install missing package {missing_pkg}: {e}")

#         if attempt >= MAX_RETRIES - 1:
#             break

#         logger.info("--- Asking LLM to Fix Code ---")
#         new_code = await fix_code_with_llm(code, result["stderr"])
#         if new_code:
#             code = new_code
#         else:
#             logger.error(
#                 "LLM failed to provide a valid code fix. Retrying with original code."
#             )

#         attempt += 1

#     raise HTTPException(
#         status_code=500, detail="Code failed after multiple LLM fix attempts."
#     )
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
    match = re.search(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_dependencies(code: str) -> list[str]:
    """Extract only the dependencies explicitly listed by the LLM."""
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", code, re.DOTALL)
    if match:
        deps_str = match.group(1)
        # Handle multi-line dependencies
        deps = []
        for item in re.findall(r'["\']([^"\']+)["\']', deps_str):
            deps.append(item.strip())
        return deps
    return []


def install_dependencies(deps: list[str]):
    """Install only the dependencies explicitly specified by the LLM."""
    if not deps:
        logger.info("No external dependencies specified.")
        return

    logger.info(f"Installing LLM-specified dependencies: {deps}")
    for dep in deps:
        logger.info(f"Installing {dep}...")
        try:
            result = subprocess.run(
                ["uv", "pip", "install", dep],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                logger.warning(f"'uv' failed for {dep}. Trying pip...")
                pip_result = subprocess.run(
                    ["pip", "install", dep],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if pip_result.returncode != 0:
                    logger.error(
                        f"Both uv and pip failed to install {dep}.\n"
                        f"uv stderr: {result.stderr.decode()}\n"
                        f"pip stderr: {pip_result.stderr.decode()}"
                    )
                    raise Exception(f"Failed to install dependency: {dep}")
        except Exception as e:
            logger.error(f"Error installing {dep}: {e}")
            raise


def run_code_with_uv(code: str) -> dict:
    try:
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
    api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM3NTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.gd84YB3ekhIQzU4E4Tgogaeya9idpV6hMTi6uPja7j0"
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
    fix_prompt = FIX_CODE_PROMPT_TEMPLATE.replace("{code}", code).replace(
        "{error}", error
    )
    response = await call_llm(CODER_SYSTEM_PROMPT, fix_prompt)
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
    logger.info(f"--- Plan Generated ---\n{plan}")

    logger.info("--- Generating Code ---")
    code_block = await call_llm(CODER_SYSTEM_PROMPT, plan)
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
            # Only install dependencies explicitly listed by the LLM
            deps = extract_dependencies(code)
            install_dependencies(deps)
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Dependency installation failed: {e}"
            )

        logger.info("--- Running Code ---")
        result = run_code_with_uv(code)

        # If execution succeeded, return output
        if result["success"]:
            logger.info(f"--- Code Execution Successful ---\n{result['stdout']}")
            return {"output": result["stdout"]}

        logger.warning(
            f"--- Code Execution Failed ---\nStderr: {result['stderr']}\nStdout: {result['stdout']}"
        )

        if attempt >= MAX_RETRIES - 1:
            break

        logger.info("--- Asking LLM to Fix Code ---")
        # Pass both stderr and stdout for better error context
        error_context = f"STDERR:\n{result['stderr']}\n\nSTDOUT:\n{result['stdout']}"
        new_code = await fix_code_with_llm(code, error_context)
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
