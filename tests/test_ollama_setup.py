import pytest
import requests
import subprocess
import os
import time
from pathlib import Path
import json # For parsing curl output
from dotenv import load_dotenv  # For loading .env file

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
CERTS_DIR = BASE_DIR / "certs"
CLIENT_CERT_PATH = CERTS_DIR / "client.crt"
CLIENT_KEY_PATH = CERTS_DIR / "client.key"
CA_CERT_PATH = CERTS_DIR / "ca.crt"

OLLAMA_DIRECT_URL = "http://127.0.0.1:11434"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_DIRECT_URL}/api/tags"
LITELLM_DIRECT_URL = "http://localhost:4000"  # As per docker-compose.yml [cite: 176] and python-client/main.py [cite: 140]
LITELLM_DIRECT_CHAT_ENDPOINT = f"{LITELLM_DIRECT_URL}/chat/completions"
MTLS_PROXY_URL = "https://localhost:8443"
LITELLM_CHAT_ENDPOINT = f"{MTLS_PROXY_URL}/chat/completions"

TEST_MODEL_NAME = "ollama-qwen-local"
LITELLM_MASTER_KEY = "sk-1234"

MAX_READINESS_WAIT_SECONDS = 60
READINESS_CHECK_INTERVAL_SECONDS = 5



# --- Helper Functions ---

def run_command(command_list, cwd=None, check=True, capture_output=True, text=True, env_vars=None):
    """Runs a generic command using subprocess."""
    print(f"Running command: {' '.join(command_list)}")
    effective_env = {**os.environ, **(env_vars or {})}
    try: # [cite: 13]
        process = subprocess.run(
            command_list,
            cwd=cwd or BASE_DIR,
            capture_output=capture_output,
            text=text,
            check=check,
            env=effective_env
        )
        if capture_output: # [cite: 14]
            if process.stdout and process.stdout.strip():
                print(f"Command stdout:\n{process.stdout.strip()}")
            if process.stderr and process.stderr.strip():
                print(f"Command stderr:\n{process.stderr.strip()}")
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command_list)}")
        print(f"Return code: {e.returncode}") # [cite: 15]
        if capture_output:
            if e.stdout and e.stdout.strip():
                print(f"Stdout:\n{e.stdout.strip()}")
            if e.stderr and e.stderr.strip():
                print(f"Stderr:\n{e.stderr.strip()}")
        if check:
            pytest.fail(f"Command failed: {' '.join(command_list)}. Error: {e.stderr or e.stdout or 'Unknown error'}") # [cite: 16]
    except FileNotFoundError:
        pytest.fail(f"Command {command_list[0]} not found. Is it installed and in PATH?")


def run_docker_compose_command(command_args, env_vars=None, check=True):
    """Wrapper for docker-compose commands."""
    return run_command(["docker-compose"] + command_args, env_vars=env_vars, check=check)


# --- Pytest Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def ensure_services_are_up():
    print("Checking if essential Docker services are running and Ollama is responsive...")
    print(f"Will wait up to {MAX_READINESS_WAIT_SECONDS} seconds for Ollama at {OLLAMA_TAGS_ENDPOINT}.")
    start_time = time.time()
    ollama_ready = False
    last_exception = None
    while time.time() - start_time < MAX_READINESS_WAIT_SECONDS: # [cite: 17]
        try:
            print(f"Attempting to connect to Ollama at {OLLAMA_TAGS_ENDPOINT} (Attempt {int((time.time() - start_time) / READINESS_CHECK_INTERVAL_SECONDS) + 1})")
            response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
            response.raise_for_status()
            if response.status_code == 200 and "models" in response.json():
                print("Ollama is responsive and returned model list.") # [cite: 18]
                ollama_ready = True
                break
            else:
                last_exception = Exception(f"Ollama format unexpected: {response.text[:100]}")
        except requests.exceptions.RequestException as e:
            last_exception = e # [cite: 19]
            print(f"Error connecting to Ollama or unexpected response. Retrying... Error: {e}") # [cite: 20]
        time.sleep(READINESS_CHECK_INTERVAL_SECONDS)
    if not ollama_ready:
        pytest.fail(
            f"Ollama service at {OLLAMA_TAGS_ENDPOINT} did not become responsive "
            f"within {MAX_READINESS_WAIT_SECONDS} seconds. Last error: {last_exception}\n"
            "Ensure services are up: 'docker-compose up -d ollama litellm-proxy nginx-mtls-proxy'. "
            "Check logs: 'docker-compose logs ollama'."
        ) # [cite: 21]
    print("Ollama confirmed ready. Giving a brief moment for other services...")
    time.sleep(5)

@pytest.fixture(scope="module")
def regenerate_certificates_and_restart_nginx():
    print("Regenerating certificates with SAN for server and restarting Nginx...")
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    # Rebuild cert-generator if Dockerfile changed (e.g., new openssl_server.cnf)
    run_docker_compose_command(["build", "cert-generator"])
    run_docker_compose_command(["run", "--rm", "-e", "FORCE_REGENERATE=true", "cert-generator"])
    print("Certificates regenerated.")
    expected_certs = [CA_CERT_PATH, CLIENT_CERT_PATH, CLIENT_KEY_PATH, CERTS_DIR / "server.crt", CERTS_DIR / "server.key"]
    if any(not cert.exists() for cert in expected_certs):
        pytest.fail(f"Missing cert files after generation: {[str(c) for c in expected_certs if not c.exists()]}") # [cite: 22]
    print("Restarting Nginx proxy...")
    run_docker_compose_command(["restart", "nginx-mtls-proxy"])
    print("Nginx restarted. Allowing a moment for it to initialize...") # [cite: 23]
    time.sleep(5)
    return True

# --- Test Cases ---

def test_ollama_direct_accessible():
    """Test 1: Check if Ollama service is directly accessible."""
    print(f"Testing direct Ollama access at {OLLAMA_TAGS_ENDPOINT}")
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=10)
        response.raise_for_status()
        assert "models" in response.json(), "Ollama /api/tags response invalid"
        print("Ollama direct access successful.")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Failed to connect to Ollama directly at {OLLAMA_TAGS_ENDPOINT}: {e}") # [cite: 24]

@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
def test_certificates_are_present_after_regeneration():
    """Test 2: Verify certificate files are present after regeneration."""
    print("Verifying presence of generated certificate files...")
    assert CA_CERT_PATH.exists(), f"CA certificate missing: {CA_CERT_PATH}"
    assert CLIENT_CERT_PATH.exists(), f"Client certificate missing: {CLIENT_CERT_PATH}"
    assert (CERTS_DIR / "server.crt").exists(), f"Server certificate missing"
    print("All expected certificate files are present.")


class TestLiteLLMDirect:
    def test_litellm_direct_correct_auth(self):
        """Test accessing LiteLLM proxy directly with correct authentication."""
        print(f"Testing LiteLLM direct access with correct auth: {LITELLM_DIRECT_CHAT_ENDPOINT}")
        payload = {
            "model": TEST_MODEL_NAME, # Should be "ollama-qwen-local" as per your litellm_proxy/config.yaml
            "messages": [{"role": "user", "content": "What is the capital of Germany?"}],
            "max_tokens": 10
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LITELLM_MASTER_KEY}" # [cite: 155, 158, 161, 163, 168, 172]
        }
        try:
            response = requests.post(
                LITELLM_DIRECT_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()  # Raises an exception for 4XX/5XX status codes
            assert "choices" in response.json(), "Response missing 'choices'"
            print("LiteLLM direct access with correct auth successful.")
            # You can add more assertions here, e.g., checking the content of the response
            # print(f"Response: {response.json()}")
        except requests.exceptions.RequestException as e:
            # If litellm-proxy is not running on port 4000, this might be a ConnectionError
            pytest.fail(f"Request to LiteLLM proxy failed: {e}")

    def test_litellm_direct_no_auth(self):
        """Test accessing LiteLLM proxy directly without authentication."""
        print(f"Testing LiteLLM direct access with NO auth: {LITELLM_DIRECT_CHAT_ENDPOINT}")
        payload = {
            "model": TEST_MODEL_NAME,
            "messages": [{"role": "user", "content": "Test no auth."}],
            "max_tokens": 5
        }
        headers = {"Content-Type": "application/json"} # No Authorization header
        try:
            response = requests.post(
                LITELLM_DIRECT_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=10
            )
            # LiteLLM proxy typically returns 401 Unauthorized if master_key is set and not provided
            assert response.status_code == 401, f"Expected HTTP 401 Unauthorized, got {response.status_code}. Response: {response.text}"
            print("LiteLLM direct access without auth correctly failed with HTTP 401.")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Unexpected exception with no auth: {e}")

    def test_litellm_direct_incorrect_auth(self):
        """Test accessing LiteLLM proxy directly with incorrect authentication."""
        print(f"Testing LiteLLM direct access with INCORRECT auth: {LITELLM_DIRECT_CHAT_ENDPOINT}")
        payload = {
            "model": TEST_MODEL_NAME,
            "messages": [{"role": "user", "content": "Test incorrect auth."}],
            "max_tokens": 5
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer wrong-key" # Incorrect key
        }
        try:
            response = requests.post(
                LITELLM_DIRECT_CHAT_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=10
            )
            assert response.status_code == 401, f"Expected HTTP 401 Unauthorized, got {response.status_code}. Response: {response.text}"
            print("LiteLLM direct access with incorrect auth correctly failed with HTTP 401.")
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Unexpected exception with incorrect auth: {e}")

# --- Requests-based mTLS Tests ---
@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
class TestMTLSWithRequests:
    def test_ollama_via_mtls_proxy_correct_cert_requests(self):
        """Test 3a: (Requests) Access Ollama via mTLS proxy with correct client certificate.""" # [cite: 25]
        print(f"Testing (Requests) Ollama via mTLS proxy with correct cert: {LITELLM_CHAT_ENDPOINT}")
        payload = {"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "Capital of France?"}], "max_tokens": 5}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LITELLM_MASTER_KEY}"}
        try:
            response = requests.post(
                LITELLM_CHAT_ENDPOINT, json=payload, headers=headers,
                cert=(str(CLIENT_CERT_PATH), str(CLIENT_KEY_PATH)), # [cite: 26]
                verify=str(CA_CERT_PATH), timeout=30
            )
            response.raise_for_status()
            assert "choices" in response.json(), "Response missing 'choices'"
            print("(Requests) mTLS access with correct cert successful.")
        except requests.exceptions.SSLError as e:
            pytest.fail(f"(Requests) SSL error with correct certs: {e}. Nginx logs: 'docker-compose logs nginx-mtls-proxy'") # [cite: 27, 28]
        except requests.exceptions.RequestException as e:
            pytest.fail(f"(Requests) Request failed with correct certs: {e}")

    def test_ollama_via_mtls_proxy_no_client_cert_requests(self):
        """Test 4a: (Requests) Attempt mTLS proxy WITHOUT client certificate."""
        print(f"Testing (Requests) Ollama via mTLS proxy with NO client cert: {LITELLM_CHAT_ENDPOINT}")
        payload = {"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "Test no client cert."}], "max_tokens": 5}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LITELLM_MASTER_KEY}"} # [cite: 29]
        try:
            response = requests.post(LITELLM_CHAT_ENDPOINT, json=payload, headers=headers, verify=str(CA_CERT_PATH), timeout=10)
            assert response.status_code == 400, f"Expected HTTP 400, got {response.status_code}. Response: {response.text}" # [cite: 30]
            print("(Requests) mTLS access without client cert correctly failed with HTTP 400.")
        except requests.exceptions.SSLError:
            print("(Requests) mTLS access without client cert correctly failed with SSLError.")
            assert True
        except requests.exceptions.RequestException as e:
            pytest.fail(f"(Requests) Unexpected exception with no client cert: {e}")

    def test_ollama_via_mtls_proxy_server_not_trusted_by_client_requests(self): # [cite: 31]
        """Test 5a: (Requests) Attempt mTLS proxy when client uses default CAs (server should not be trusted)."""
        print(f"Testing (Requests) mTLS proxy with client cert, client NOT trusting server's custom CA: {LITELLM_CHAT_ENDPOINT}")
        payload = {"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "Test server not trusted."}], "max_tokens": 5}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LITELLM_MASTER_KEY}"}
        with pytest.raises(requests.exceptions.SSLError) as excinfo:
            requests.post( # [cite: 32]
                LITELLM_CHAT_ENDPOINT, json=payload, headers=headers,
                cert=(str(CLIENT_CERT_PATH), str(CLIENT_KEY_PATH)),
                verify=True, timeout=10 # verify=True uses system CAs
            )
        print(f"(Requests) mTLS with untrusted server CA correctly failed: {excinfo.value}")
        assert "certificate verify failed" in str(excinfo.value).lower()

# --- Curl-based mTLS Tests --- # [cite: 33]
@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
class TestMTLSWithCurl:
    def test_ollama_via_mtls_proxy_correct_cert_curl(self):
        """Test 3b: (curl) Access Ollama via mTLS proxy with correct client certificate."""
        print(f"Testing (curl) Ollama via mTLS proxy with correct cert: {LITELLM_CHAT_ENDPOINT}")
        payload = json.dumps({"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "Capital of France using curl?"}], "max_tokens": 5})
        curl_command = [
            "curl", "-s", "-X", "POST", LITELLM_CHAT_ENDPOINT,
            "--cert", str(CLIENT_CERT_PATH), # [cite: 34]
            "--key", str(CLIENT_KEY_PATH),
            "--cacert", str(CA_CERT_PATH),
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {LITELLM_MASTER_KEY}",
            "-d", payload
        ]
        process = run_command(curl_command, check=False) # Don't fail test immediately
        assert process.returncode == 0, f"curl command failed with exit code {process.returncode}. Stderr: {process.stderr}" # [cite: 35, 36]
        try:
            response_data = json.loads(process.stdout)
            assert "choices" in response_data, f"curl response missing 'choices'. Output: {process.stdout}" # [cite: 37]
            print("(curl) mTLS access with correct cert successful.")
        except json.JSONDecodeError:
            pytest.fail(f"curl output was not valid JSON: {process.stdout}")

    def test_ollama_via_mtls_proxy_no_client_cert_curl(self):
        """Test 4b: (curl) Attempt mTLS proxy WITHOUT client certificate."""
        print(f"Testing (curl) Ollama via mTLS proxy with NO client cert: {LITELLM_CHAT_ENDPOINT}")
        payload = json.dumps({"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "Test no client cert curl."}], "max_tokens": 5}) # [cite: 38]
        curl_command = [ # [cite: 39]
            "curl", "-s", "-X", "POST", LITELLM_CHAT_ENDPOINT,
            # No --cert or --key
            "--cacert", str(CA_CERT_PATH), # Still try to verify server if connection is made
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {LITELLM_MASTER_KEY}",
            "-d", payload, # [cite: 40]
            "--fail-with-body" # Makes curl exit with 22 on 4xx/5xx errors, easier to check
        ]
        process = run_command(curl_command, check=False) # Don't fail test immediately

        if process.returncode != 0: # [cite: 41]
            print(f"(curl) mTLS access without client cert correctly failed with exit code {process.returncode}. Stderr: {process.stderr}") # [cite: 42]
            assert process.returncode in [22, 35, 56, 60], f"Expected curl to fail (e.g. exit 22, 35, 56, 60), got {process.returncode}"
        else:
            pytest.fail(f"(curl) mTLS access without client cert unexpectedly succeeded (exit 0). Output: {process.stdout}") # [cite: 43]

    def test_ollama_via_mtls_proxy_server_not_trusted_by_client_curl(self):
        """Test 5b: (curl) Attempt mTLS proxy when client uses default CAs (server should not be trusted).""" # [cite: 44]
        print(f"Testing (curl) mTLS proxy with client cert, client NOT trusting server's custom CA: {LITELLM_CHAT_ENDPOINT}")
        payload = json.dumps({"model": TEST_MODEL_NAME, "messages": [{"role": "user", "content": "Test server not trusted curl."}], "max_tokens": 5})
        curl_command = [
            "curl", "-s", "-X", "POST", LITELLM_CHAT_ENDPOINT,
            "--cert", str(CLIENT_CERT_PATH),
            "--key", str(CLIENT_KEY_PATH),
            # NO --cacert str(CA_CERT_PATH) means curl uses its default CA bundle # [cite: 45]
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {LITELLM_MASTER_KEY}",
            "-d", payload
        ]
        process = run_command(curl_command, check=False)
        assert process.returncode == 60, \
            f"(curl) Expected SSL verification failure (exit code 60), got {process.returncode}. Stderr: {process.stderr}, Stdout: {process.stdout}" # [cite: 46]
        print(f"(curl) mTLS with untrusted server CA correctly failed with exit code {process.returncode}.")