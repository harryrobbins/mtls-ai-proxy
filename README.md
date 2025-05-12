# Local mTLS Testing Environment for LLM Services

This setup uses Docker Compose to create a local environment with:
1. An Nginx reverse proxy enforcing mTLS.
2. An Ollama service running a local LLM (e.g., `qwen2:0.5b` by default).
3. A LiteLLM Proxy service routing requests to Ollama.
4. A utility container to generate self-signed mTLS certificates.

This allows testing of clients (like a custom LiteLLM provider) that need to communicate with a backend service requiring mTLS.

## Prerequisites

- Docker
- Docker Compose
- `curl` (for testing)

## Setup and Usage

### 1. Generate mTLS Certificates (One-time or if certs expire/change)

The `cert-generator` service in `docker-compose.yml` is responsible for this. The generated certificates will be placed in the `./certs` directory on your host machine.

**To generate certificates:**

a. Ensure the `./certs` directory exists in your `local-mtls-test` project root. If not, create it:
   ```bash
   mkdir -p certs
   ```

b. Run the certificate generator service. This command will build the `cert-generator` image if it doesn't exist and then run it. The container will exit after generating the certs.
   ```bash
   docker-compose up cert-generator
   ```

   This will execute the `generate_certs.sh` script inside the container, creating:
   - `ca.crt`, `ca.key` (Certificate Authority)
   - `server.crt`, `server.key` (for the Nginx mTLS proxy)
   - `client.crt`, `client.key` (for your test client/application)

   These files will be copied to your local `./certs` directory.

c. **To force regeneration** (e.g., if you change subject names or want new certs):
   You can either:
   - Delete the contents of your local `./certs` directory and re-run `docker-compose up cert-generator`.
   - Or, uncomment the `FORCE_REGENERATE=true` environment variable in the `cert-generator` service definition within `docker-compose.yml` and then run `docker-compose up cert-generator`. Remember to comment it out again afterward if you don't want to force regeneration every time.

### 2. Start the Main Services

Once certificates are generated and present in the `./certs` directory, start the Nginx, LiteLLM Proxy, and Ollama services:

```bash
docker-compose up -d nginx-mtls-proxy litellm-proxy ollama
```

The ollama service will automatically pull the model specified by OLLAMA_PULL_MODEL in docker-compose.yml (default: qwen2:0.5b) on its first startup after a fresh build or if the model isn't cached. This might take a few minutes.

Check logs to ensure services start correctly:
```bash
docker-compose logs -f
```

Look for Uvicorn running on http://0.0.0.0:4000 from local_litellm_proxy and messages indicating Ollama has pulled the model and is listening from local_ollama. Nginx logs (local_mtls_proxy) should show it's ready.

### 3. Test mTLS Endpoint with curl

The Nginx mTLS proxy listens on https://localhost:8443. The LiteLLM Proxy uses ollama-qwen-local as the model name for the qwen2:0.5b model (or whatever model is configured in litellm_proxy/config.yaml and pulled by Ollama).

**Important**: Run these curl commands from your host machine, in the local-mtls-test directory (so the relative paths to certs/ are correct).

a) Test WITH Valid Client Certificate (Successful Case):
This command tells curl to use the client certificate and key, and to trust your custom CA for validating the server's certificate.

```bash
curl -X POST https://localhost:8443/chat/completions \
  --cert certs/client.crt \
  --key certs/client.key \
  --cacert certs/ca.crt \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "ollama-qwen-local",
    "messages": [{"role": "user", "content": "Briefly, what is the capital of France?"}]
  }'
```

Expected Output: A JSON response from the Ollama model (e.g., {"... "content": "Paris is the capital of France." ...}).

b) Test WITHOUT Client Certificate (mTLS Failure Case):
Nginx is configured to require a client certificate (ssl_verify_client on), so it should reject the TLS handshake or return an error. Adding -v gives verbose output.

```bash
curl -X POST https://localhost:8443/chat/completions \
  --cacert certs/ca.crt \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "ollama-qwen-local",
    "messages": [{"role": "user", "content": "Test no client cert."}]
  }' \
  -v
```

**Expected Output:** A TLS handshake failure. Look for messages like:
- `* OpenSSL SSL_connect: SSL_ERROR_SYSCALL in connection to localhost:8443`
- `* schannel: failed to decrypt data, data not available`
- `* NSS: client certificate not found (nickname not specified)`
- `curl: (35) ... ssl handshake failure`

Or, Nginx might return an HTTP 400 error: `<html><head><title>400 No required SSL certificate was sent</title></head>...</html>`.

c) Test WITH Client Certificate but Server Cert Not Trusted by Client (TLS Failure Case):
This simulates if your client doesn't trust the CA that signed the Nginx server's certificate. We achieve this by *not* providing `--cacert` to `curl`.

```bash
curl -X POST https://localhost:8443/chat/completions \
  --cert certs/client.crt \
  --key certs/client.key \
  # --cacert certs/ca.crt  <-- CA certificate is NOT provided to curl
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "ollama-qwen-local",
    "messages": [{"role": "user", "content": "Test server trust fail."}]
  }' \
  -v
```

Expected Output: A certificate verification error from curl because it cannot verify the Nginx server's certificate against its known CAs (since our custom CA isn't in the system's default trust store).
* SSL certificate problem: self-signed certificate in certificate chain (or similar, as our CA is self-signed and not known to curl without --cacert)
* curl: (60) SSL certificate problem...

### 4. Configure Your Main Python Application (for mTLS Provider Testing)

To test your application's AzureAPIMMTLSProvider against this local setup:

Environment Variables for your Python App:
Set these where your Python application runs (e.g., in your .env file for the main project, adjusting the paths as necessary if your main app is not in the same parent directory as local-mtls-test):

```
# Example if local-mtls-test is a sibling to your app's root
# AZURE_APIM_MTLS_CERT_PATH=../local-mtls-test/certs/client.crt
# AZURE_APIM_MTLS_KEY_PATH=../local-mtls-test/certs/client.key

# Or, if your main app is at /path/to/main-app and local-mtls-test is at /path/to/local-mtls-test
# AZURE_APIM_MTLS_CERT_PATH=/path/to/local-mtls-test/certs/client.crt
# AZURE_APIM_MTLS_KEY_PATH=/path/to/local-mtls-test/certs/client.key

# For simplicity during initial testing, you can use absolute paths:
# AZURE_APIM_MTLS_CERT_PATH=/full/path/to/your/local-mtls-test/certs/client.crt
# AZURE_APIM_MTLS_KEY_PATH=/full/path/to/your/local-mtls-test/certs/client.key
```

**Important**: Ensure these paths are resolvable from the directory where your Python application process is running.

llm_providers.json in your Python App:
Configure the provider to point to the local Nginx mTLS proxy. The provider_type must match CUSTOM_PROVIDER_NAME from custom_llm_providers.py.

```json
{
  // ... other providers ...
  "test-local-mtls-ollama": {
    "display_name": "Local mTLS Ollama (Qwen)",
    "provider_type": "azure-apim-mtls", // Your custom provider's registered name
    "deployment_name": "ollama-qwen-local", // Must match model_name in litellm_proxy/config.yaml
    "api_base": "https://localhost:8443",    // Nginx mTLS endpoint
    "api_key": "sk-1234", // The master_key for LiteLLM Proxy auth (passed to custom provider)
    "api_version": "2024-02-01" // Dummy version if your custom provider expects/needs it
  }
  // ...
}
```

Run your Python application. When you make a request using the test-local-mtls-ollama model key, your AzureAPIMMTLSProvider should handle the mTLS connection to https://localhost:8443.

### 5. Stopping the Services

When you're done testing:
```bash
docker-compose down
```

This stops and removes the containers. The `./certs` directory (and its contents) will persist on your host machine.

To also remove the Docker network if it's not needed by other Compose projects:
```bash
docker-compose down --remove-orphans
```

## Additional Notes

This environment provides a way to locally test mTLS interactions. The AzureAPIMMTLSProvider's SSL context is configured to load default CAs for server certificate verification, which works for publicly trusted CAs. For this local setup with a self-signed CA, your client application (Python with httpx) would typically need to be configured to trust certs/ca.crt if it were verifying the server cert itself directly. However, the httpx.Client(verify=ssl_context) where ssl_context only has load_cert_chain (client cert) and load_default_certs relies on those default CAs. If localhost's cert isn't signed by one of those, you might need to adjust the client's ssl_context to also trust your custom ca.crt for the server verification part, or use verify=path/to/your/ca.crt in httpx.Client. The curl example explicitly uses --cacert for this reason.

For the AzureAPIMMTLSProvider, the ssl.create_default_context(ssl.Purpose.SERVER_AUTH) combined with ssl_context.load_default_certs() should be sufficient if the server (Nginx) presents a cert verifiable by standard CAs. For our local Nginx using a custom CA, the httpx.Client(verify=ssl_context) in the custom provider will need that ssl_context to also trust our ca.crt. Let's ensure the custom provider's _create_ssl_context does this.

Refinement for AzureAPIMMTLSProvider._create_ssl_context (in your main app's custom_llm_providers.py) to trust the local CA:
If you encounter SSL verification errors when your Python app connects to https://localhost:8443 (because localhost's cert is signed by your custom CA which isn't in the default trust store), you'll need to tell httpx to trust your CA for server certificate validation.

In app/core/custom_llm_providers.py (your main application, not the test setup):

```python
# In AzureAPIMMTLSProvider, method _create_ssl_context:
# ...
    def _create_ssl_context(self) -> ssl.SSLContext:
        try:
            # For SERVER_AUTH, we need to specify the CA that signed the SERVER's cert
            # if it's not a publicly trusted one.
            # For this local test, our server (Nginx) uses a cert signed by our ca.crt
            # So, we need to load ca.crt for server verification.
            ca_for_server_verification_path = os.path.join(
                os.path.dirname(self.cert_path), # Assuming ca.crt is in the same dir as client.crt
                "ca.crt" # Or get this path from another env var / config
            )
            if not os.path.exists(ca_for_server_verification_path):
                logger.warning(f"CA certificate for server verification not found at {ca_for_server_verification_path}. SSL/TLS server verification might fail if server uses a custom CA.")
                # Fallback to default CAs only if our specific one isn't found
                ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                ssl_context.load_default_certs()
            else:
                logger.info(f"Loading custom CA for server verification: {ca_for_server_verification_path}")
                ssl_context = ssl.create_default_context(
                    ssl.Purpose.SERVER_AUTH,
                    cafile=ca_for_server_verification_path
                )

            ssl_context.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path) # Client cert
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED # Verify server's cert
            return ssl_context
        # ... (rest of the error handling)
# ...
```

This refinement to _create_ssl_context in your actual application's custom provider will be important when testing against the local Nginx server that uses a certificate signed by your custom ca.crt. You'd also need to ensure ca.crt is available to your main application (e.g., copy it from local-mtls-test/certs/ to a location your app can access).