# conftest.py
# This file can be used for project-wide fixtures, like ensuring docker services are up.

import pytest
import requests
import subprocess
import os
import time
from pathlib import Path
import json
import ssl
import httpx
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration (should match test_ollama_setup.py) ---
BASE_DIR = Path(__file__).resolve().parent.parent
CERTS_DIR = BASE_DIR / "certs"
CLIENT_CERT_PATH = CERTS_DIR / "client.crt"
CLIENT_KEY_PATH = CERTS_DIR / "client.key"
CA_CERT_PATH = CERTS_DIR / "ca.crt"

OLLAMA_DIRECT_URL = "http://127.0.0.1:11434"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_DIRECT_URL}/api/tags"
LITELLM_DIRECT_URL = "http://localhost:4000"
LITELLM_DIRECT_CHAT_ENDPOINT = f"{LITELLM_DIRECT_URL}/chat/completions"
MTLS_PROXY_URL = "https://localhost:8443"

TEST_MODEL_NAME = "ollama-qwen-local"
LITELLM_PROXY_MODEL_FOR_OPENAI_SDK = "ollama-qwen-local"
LITELLM_MASTER_KEY = "sk-1234"

MAX_READINESS_WAIT_SECONDS = 90 # Increased wait time
READINESS_CHECK_INTERVAL_SECONDS = 5

# --- Helper Functions (copied from test_ollama_setup.py) ---

def run_command(command_list, cwd=None, check=True, capture_output=True, text=True, env_vars=None):
    """Runs a generic command using subprocess."""
    print(f"\nRunning command: {' '.join(command_list)}")
    effective_env = {**os.environ, **(env_vars or {})}
    try:
        process = subprocess.run(
            command_list,
            cwd=cwd or BASE_DIR,
            capture_output=capture_output,
            text=text,
            check=check,
            env=effective_env
        )
        if capture_output:
            if process.stdout and process.stdout.strip():
                print(f"Command stdout:\n{process.stdout.strip()}")
            if process.stderr and process.stderr.strip():
                print(f"Command stderr:\n{process.stderr.strip()}")
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command_list)}")
        print(f"Return code: {e.returncode}")
        if capture_output:
            if e.stdout and e.stdout.strip():
                print(f"Stdout:\n{e.stdout.strip()}")
            if e.stderr and e.stderr.strip():
                print(f"Stderr:\n{e.stderr.strip()}")
        if check:
            pytest.fail(f"Command failed: {' '.join(command_list)}. Error: {e.stderr or e.stdout or 'Unknown error'}")
    except FileNotFoundError:
        pytest.fail(f"Command {command_list[0]} not found. Is it installed and in PATH?")


def run_docker_compose_command(command_args, env_vars=None, check=True):
    """Wrapper for docker-compose commands."""
    # Add -f flag to explicitly specify the docker-compose file
    return run_command(["docker-compose", "-f", str(BASE_DIR / "docker-compose.yml")] + command_args, env_vars=env_vars, check=check)


# --- Project-wide Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def ensure_services_are_up_and_certs_generated():
    """
    Ensures necessary Docker services are up and certificates are generated
    before running any tests in the session.
    """
    print("\n--- Ensuring Docker Services are Up and Certificates are Generated ---")

    # 1. Regenerate certificates
    print("Attempting to regenerate certificates...")
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Use FORCE_REGENERATE=true to ensure fresh certs
        run_docker_compose_command(["run", "--rm", "-e", "FORCE_REGENERATE=true", "cert-generator"])
        print("Certificates regenerated.")
        expected_certs = [CA_CERT_PATH, CLIENT_CERT_PATH, CLIENT_KEY_PATH, CERTS_DIR / "server.crt", CERTS_DIR / "server.key"]
        if any(not cert.exists() for cert in expected_certs):
             pytest.fail(f"Missing cert files after generation: {[str(c) for c in expected_certs if not c.exists()]}")
        print("All expected certificate files are present.")
    except Exception as e:
         pytest.fail(f"Failed to regenerate certificates: {e}")


    # 2. Start the main services (ollama, litellm-proxy, nginx-mtls-proxy)
    print("Starting essential Docker services...")
    try:
        # Use --wait to wait for services to be healthy (if healthchecks are defined)
        # Or rely on manual readiness checks below
        run_docker_compose_command(["up", "-d", "--build", "ollama", "litellm-proxy", "nginx-mtls-proxy"])
        print("Docker services started (ollama, litellm-proxy, nginx-mtls-proxy).")
    except Exception as e:
         pytest.fail(f"Failed to start docker services: {e}")


    # 3. Wait for Ollama to be responsive (Manual readiness check)
    print(f"Waiting up to {MAX_READINESS_WAIT_SECONDS} seconds for Ollama at {OLLAMA_TAGS_ENDPOINT}...")
    start_time = time.time()
    ollama_ready = False
    last_exception = None

    while time.time() - start_time < MAX_READINESS_WAIT_SECONDS:
        try:
            # Use requests directly to check Ollama endpoint
            response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            if response.status_code == 200 and "models" in response.json():
                print("Ollama is responsive and returned model list.")
                ollama_ready = True
                break
            else:
                last_exception = Exception(f"Ollama format unexpected: {response.text[:100]}")
        except requests.exceptions.RequestException as e:
            last_exception = e
            print(f"Error connecting to Ollama or unexpected response. Retrying... Error: {type(e).__name__}: {e}")
        time.sleep(READINESS_CHECK_INTERVAL_SECONDS)

    if not ollama_ready:
        pytest.fail(
            f"Ollama service at {OLLAMA_TAGS_ENDPOINT} did not become responsive "
            f"within {MAX_READINESS_WAIT_SECONDS} seconds. Last error: {last_exception}\n"
            "Ensure services are up: 'docker-compose up -d ollama litellm-proxy nginx-mtls-proxy'. "
            "Check logs: 'docker-compose logs ollama litellm-proxy nginx-mtls-proxy'."
        )

    print("Ollama confirmed ready. Giving a brief moment for other services (LiteLLM, Nginx)...")
    # Give LiteLLM and Nginx a little more time to fully initialize and load configs/certs
    time.sleep(10)
    print("Setup complete. Running tests.")

    # Yield control to the tests
    yield

    # Teardown: Stop the Docker services after all tests are done
    print("\n--- Tearing down Docker Services ---")
    try:
        run_docker_compose_command(["down", "--remove-orphans"])
        print("Docker services stopped.")
    except Exception as e:
        print(f"Warning: Failed to stop docker services cleanly: {e}")