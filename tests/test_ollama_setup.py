import pytest
import requests
import subprocess
import os
import time
from pathlib import Path

# --- Configuration ---
# Base directory of the project (assuming tests are in a 'tests' subdirectory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Certs directory
CERTS_DIR = BASE_DIR / "certs"
CLIENT_CERT_PATH = CERTS_DIR / "client.crt"
CLIENT_KEY_PATH = CERTS_DIR / "client.key"
CA_CERT_PATH = CERTS_DIR / "ca.crt"

# Service URLs
OLLAMA_DIRECT_URL = "http://127.0.0.1:11434"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_DIRECT_URL}/api/tags"
MTLS_PROXY_URL = "https://localhost:8443"
LITELLM_CHAT_ENDPOINT = f"{MTLS_PROXY_URL}/chat/completions"

# Model to use for testing (should match litellm_proxy/config.yaml)
TEST_MODEL_NAME = "ollama-qwen-local"
LITELLM_MASTER_KEY = "sk-1234"  # As per litellm_proxy/config.yaml

# Readiness check parameters
MAX_READINESS_WAIT_SECONDS = 60
READINESS_CHECK_INTERVAL_SECONDS = 5


# --- Helper Functions ---

def run_docker_compose_command(command_args, env_vars=None, check=True):
    """Runs a docker-compose command."""
    cmd = ["docker-compose"] + command_args
    print(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=check,  # Allow disabling check for commands that might fail (e.g., down)
            env={**os.environ, **(env_vars or {})}
        )
        if process.stdout:  # Print stdout only if it's not empty
            print(f"Command stdout:\n{process.stdout.strip()}")
        if process.stderr:  # Print stderr only if it's not empty
            print(f"Command stderr:\n{process.stderr.strip()}")
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout.strip()}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr.strip()}")
        if check:  # Only fail the test if check=True
            pytest.fail(f"Docker-compose command failed: {' '.join(cmd)}. Error: {e.stderr or e.stdout}")
    except FileNotFoundError:
        pytest.fail("docker-compose command not found. Is it installed and in PATH?")


# --- Pytest Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def ensure_services_are_up():
    """
    Fixture to ensure main services, especially Ollama, are up and responsive.
    Retries checking the Ollama direct endpoint for a specified duration.
    """
    print("Checking if essential Docker services are running and Ollama is responsive...")
    print(f"Will wait up to {MAX_READINESS_WAIT_SECONDS} seconds for Ollama at {OLLAMA_TAGS_ENDPOINT}.")
    print("Ensure 'docker-compose up -d nginx-mtls-proxy litellm-proxy ollama' has been run.")

    start_time = time.time()
    ollama_ready = False
    last_exception = None

    while time.time() - start_time < MAX_READINESS_WAIT_SECONDS:
        try:
            print(
                f"Attempting to connect to Ollama at {OLLAMA_TAGS_ENDPOINT} (Attempt {int((time.time() - start_time) / READINESS_CHECK_INTERVAL_SECONDS) + 1})")
            response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
            response.raise_for_status()
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "models" in data:
                        print("Ollama is responsive and returned model list.")
                        ollama_ready = True
                        break
                    else:
                        print(f"Ollama responded, but /api/tags format unexpected: {data}")
                        last_exception = Exception(f"Ollama /api/tags format unexpected: {data}")
                except ValueError:  # Handles cases where response is not JSON
                    print(f"Ollama responded to {OLLAMA_TAGS_ENDPOINT}, but not with valid JSON: {response.text[:100]}")
                    last_exception = Exception(f"Ollama responded to {OLLAMA_TAGS_ENDPOINT}, but not with valid JSON.")
            # else: # Should be caught by raise_for_status, but as a fallback
            #     print(f"Ollama at {OLLAMA_TAGS_ENDPOINT} responded with status {response.status_code}.")
            #     last_exception = requests.exceptions.HTTPError(f"Status {response.status_code}")

        except requests.exceptions.ConnectionError as e:
            print(f"Ollama not yet reachable at {OLLAMA_TAGS_ENDPOINT}. Retrying... Error: {e}")
            last_exception = e
        except requests.exceptions.RequestException as e:  # Catch other request-related errors like HTTP errors
            print(
                f"Error connecting to Ollama or unexpected response from {OLLAMA_TAGS_ENDPOINT}. Retrying... Error: {e}")
            last_exception = e

        time.sleep(READINESS_CHECK_INTERVAL_SECONDS)

    if not ollama_ready:
        failure_message = (
            f"Ollama service at {OLLAMA_TAGS_ENDPOINT} did not become responsive "
            f"within {MAX_READINESS_WAIT_SECONDS} seconds. Last error: {last_exception}\n"
            "Please ensure 'ollama', 'litellm-proxy', and 'nginx-mtls-proxy' services are running "
            "and healthy: 'docker-compose up -d ollama litellm-proxy nginx-mtls-proxy'.\n"
            "Check 'docker-compose logs ollama' for issues.\n"
            "Troubleshooting steps:\n"
            "1. Verify Ollama port mapping: `docker ps` (should show 11434 mapped).\n"
            "2. Test with curl from host: `curl http://127.0.0.1:11434/api/tags`.\n"
            "3. Check for proxy settings (HTTP_PROXY, HTTPS_PROXY, NO_PROXY) in your test environment.\n"
            "4. Check host firewall rules.\n"
            "5. If using WSL, check WSL networking and port forwarding."
        )
        pytest.fail(failure_message)

    print("Ollama confirmed ready. Giving a brief moment for other services...")
    time.sleep(5)  # Give other services a moment to be fully ready after Ollama


@pytest.fixture(scope="module")
def regenerate_certificates_and_restart_nginx():
    """
    Fixture to ensure certificates are freshly generated and Nginx is restarted.
    This runs once per test module that uses it.
    """
    print("Regenerating certificates...")
    CERTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure ./certs directory exists

    # Force regenerate certificates
    run_docker_compose_command(
        ["run", "--rm", "-e", "FORCE_REGENERATE=true", "cert-generator"]
    )
    print("Certificates regenerated.")

    # Verify all expected cert files are created on the host
    expected_certs = [
        CERTS_DIR / "ca.crt", CERTS_DIR / "ca.key",
        CERTS_DIR / "server.crt", CERTS_DIR / "server.key",
        CLIENT_CERT_PATH, CLIENT_KEY_PATH
    ]
    missing_certs = [cert for cert in expected_certs if not cert.exists()]
    if missing_certs:
        pytest.fail(f"Missing certificate files in host './certs' directory after generation: {missing_certs}")

    print("Restarting Nginx proxy to load new certificates...")
    run_docker_compose_command(["restart", "nginx-mtls-proxy"])
    print("Nginx proxy restarted. Allowing a moment for it to initialize...")
    time.sleep(5)  # Give Nginx a few seconds to fully restart and load certs

    return True


# --- Test Cases ---

def test_ollama_direct_accessible():
    """Test 1: Check if Ollama service is directly accessible on its mapped port."""
    print(f"Testing direct Ollama access at {OLLAMA_TAGS_ENDPOINT}")
    try:
        response = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=10)
        response.raise_for_status()
        assert response.status_code == 200
        assert "models" in response.json(), "Ollama /api/tags response should contain 'models' key"
        print("Ollama direct access successful.")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Failed to connect to Ollama directly at {OLLAMA_TAGS_ENDPOINT}: {e}")


@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
def test_certificates_are_present_after_regeneration():
    """Test 2: Verify certificate files are present after regeneration and Nginx restart."""
    print("Verifying presence of generated certificate files (post-regeneration)...")
    assert CA_CERT_PATH.exists(), f"CA certificate missing: {CA_CERT_PATH}"
    assert (CERTS_DIR / "ca.key").exists(), f"CA key missing: {CERTS_DIR / 'ca.key'}"
    assert (CERTS_DIR / "server.crt").exists(), f"Server certificate missing: {CERTS_DIR / 'server.crt'}"
    assert (CERTS_DIR / "server.key").exists(), f"Server key missing: {CERTS_DIR / 'server.key'}"
    assert CLIENT_CERT_PATH.exists(), f"Client certificate missing: {CLIENT_CERT_PATH}"
    assert CLIENT_KEY_PATH.exists(), f"Client key missing: {CLIENT_KEY_PATH}"
    print("All expected certificate files are present after regeneration.")


@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
def test_ollama_via_mtls_proxy_correct_cert():
    """Test 3: Access Ollama via mTLS proxy using the correct client certificate."""
    print(f"Testing Ollama via mTLS proxy with correct client cert: {LITELLM_CHAT_ENDPOINT}")
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "What is the capital of France? Briefly."}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LITELLM_MASTER_KEY}"
    }
    try:
        response = requests.post(
            LITELLM_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            cert=(str(CLIENT_CERT_PATH), str(CLIENT_KEY_PATH)),  # Client cert and key
            verify=str(CA_CERT_PATH),  # CA cert to verify Nginx's server cert
            timeout=30
        )
        response.raise_for_status()  # Raise an exception for HTTP errors 4xx/5xx
        assert response.status_code == 200
        response_data = response.json()
        assert "choices" in response_data, "Response should contain 'choices'"
        assert len(response_data["choices"]) > 0, "Should have at least one choice"
        assert "message" in response_data["choices"][0], "Choice should contain 'message'"
        print("Ollama access via mTLS proxy with correct cert successful.")
        print(f"Response: {response.text[:200]}...")
    except requests.exceptions.SSLError as e:
        pytest.fail(f"SSL error accessing mTLS proxy with correct certs: {e}. "
                    "Ensure certs are correctly generated and Nginx is using them. "
                    "Check Nginx logs ('docker-compose logs nginx-mtls-proxy'). "
                    f"CA_CERT_PATH: {CA_CERT_PATH}, CLIENT_CERT_PATH: {CLIENT_CERT_PATH}")
    except requests.exceptions.RequestException as e:  # Catch other errors like connection or HTTP errors
        pytest.fail(f"Request failed for mTLS proxy with correct certs: {e}. Check Nginx, LiteLLM, and Ollama logs.")


@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
def test_ollama_via_mtls_proxy_no_client_cert():
    """Test 4: Attempt to access Ollama via mTLS proxy WITHOUT a client certificate."""
    print(f"Testing Ollama via mTLS proxy with NO client cert: {LITELLM_CHAT_ENDPOINT}")
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "Test no client cert."}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LITELLM_MASTER_KEY}"
    }
    try:
        response = requests.post(
            LITELLM_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            # No client cert provided
            verify=str(CA_CERT_PATH),  # Still provide CA to verify server if connection gets that far
            timeout=10
        )
        # Nginx should reject the connection or return a 400 error
        assert response.status_code == 400, \
            f"Expected HTTP 400 (No required SSL certificate) but got {response.status_code}. Response: {response.text}"
        print(f"Ollama access via mTLS proxy without client cert correctly failed with HTTP 400.")

    except requests.exceptions.SSLError as e:
        # This is also an expected outcome, as the SSL handshake might fail before HTTP status.
        print(f"Ollama access via mTLS proxy without client cert correctly failed with SSLError: {e}")
        assert True  # Test passes if SSLError occurs
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Unexpected request exception when testing with no client cert: {e}")


@pytest.mark.usefixtures("regenerate_certificates_and_restart_nginx")
def test_ollama_via_mtls_proxy_server_not_trusted_by_client():
    """
    Test 5: Attempt to access Ollama via mTLS proxy when client uses default CAs
    (which won't include our custom CA), so server verification should fail.
    """
    print(
        f"Testing Ollama via mTLS proxy with client cert but client NOT trusting server's custom CA: {LITELLM_CHAT_ENDPOINT}")
    payload = {
        "model": TEST_MODEL_NAME,
        "messages": [{"role": "user", "content": "Test server not trusted by client."}]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LITELLM_MASTER_KEY}"
    }
    # This will make requests use the system's default CA bundle.
    # Since our custom CA (ca.crt) signed the server.crt, and our custom CA
    # is not in the system's default bundle, SSL verification should fail.
    with pytest.raises(requests.exceptions.SSLError) as excinfo:
        requests.post(
            LITELLM_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            cert=(str(CLIENT_CERT_PATH), str(CLIENT_KEY_PATH)),
            verify=True,  # Use system default CAs for server verification
            timeout=10
        )
    print(
        f"Ollama access via mTLS proxy with client not trusting server's custom CA correctly failed with SSLError: {excinfo.value}")
    error_message = str(excinfo.value).lower()
    # Common phrases in SSL verification errors when CA is unknown
    assert "certificate verify failed" in error_message or \
           "self-signed certificate in certificate chain" in error_message or \
           "unable to get local issuer certificate" in error_message, \
        f"Expected SSL verification error due to untrusted CA, but got: {excinfo.value}"

