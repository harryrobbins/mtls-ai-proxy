#!/bin/sh
set -e

# Output directory inside the container, mapped from host's ./certs
OUTPUT_DIR="/certs_output"
CONFIG_DIR="/certs" # Directory where OpenSSL config files are copied in Dockerfile

# Check if certificates already exist to avoid overwriting unless forced
if [ -f "$OUTPUT_DIR/ca.crt" ] && [ "$FORCE_REGENERATE" != "true" ]; then
  echo "Certificates already exist in $OUTPUT_DIR. Skipping generation."
  echo "Set FORCE_REGENERATE=true to overwrite."
  exit 0
fi

echo "Generating new certificates in $CONFIG_DIR, will be copied to $OUTPUT_DIR ..."

# --- CA Certificate ---
echo "1. Generating CA private key (ca.key)..."
openssl genrsa -out "$CONFIG_DIR/ca.key" 2048

echo "2. Generating CA certificate (ca.crt) using $CONFIG_DIR/openssl_ca.cnf ..."
openssl req -x509 -new -nodes -key "$CONFIG_DIR/ca.key" \
  -sha256 -days 3650 \
  -subj "/C=XX/ST=Testland/L=TestCity/O=LocalTestOrg/OU=DevOps/CN=LocalTestCA" \
  -config "$CONFIG_DIR/openssl_ca.cnf" \
  -extensions v3_ca \
  -out "$CONFIG_DIR/ca.crt"

# --- Server Certificate ---
echo "3. Generating Server private key (server.key)..."
openssl genrsa -out "$CONFIG_DIR/server.key" 2048

echo "4. Generating Server Certificate Signing Request (server.csr) with SAN using $CONFIG_DIR/openssl_server.cnf ..."
# Pass the config file to req to include extensions (like SAN) in the CSR itself
openssl req -new -key "$CONFIG_DIR/server.key" \
  -subj "/C=XX/ST=Testland/L=TestCity/O=LocalTestOrg/OU=Server/CN=localhost" \
  -config "$CONFIG_DIR/openssl_server.cnf" \
  -reqexts v3_req \
  -out "$CONFIG_DIR/server.csr"

echo "5. Signing the Server CSR with the CA (server.crt)..."
# When signing, explicitly copy extensions from the CSR to the certificate
openssl x509 -req -in "$CONFIG_DIR/server.csr" \
  -CA "$CONFIG_DIR/ca.crt" -CAkey "$CONFIG_DIR/ca.key" \
  -CAcreateserial \
  -days 3600 -sha256 \
  -copy_extensions copyall \
  -out "$CONFIG_DIR/server.crt"

# --- Client Certificate ---
echo "6. Generating Client private key (client.key)..."
openssl genrsa -out "$CONFIG_DIR/client.key" 2048

echo "7. Generating Client Certificate Signing Request (client.csr)..."
# For client certs, SAN is less common unless specific use cases demand it.
# We'll keep it simple for now. CN=TestClient should be sufficient for client auth.
openssl req -new -key "$CONFIG_DIR/client.key" \
  -subj "/C=XX/ST=Testland/L=TestCity/O=LocalTestOrg/OU=Client/CN=TestClient" \
  -out "$CONFIG_DIR/client.csr"

echo "8. Signing the Client CSR with the CA (client.crt)..."
openssl x509 -req -in "$CONFIG_DIR/client.csr" \
  -CA "$CONFIG_DIR/ca.crt" -CAkey "$CONFIG_DIR/ca.key" \
  -CAcreateserial \
  -days 3600 -sha256 \
  -out "$CONFIG_DIR/client.crt"
  # Add extensions for client if needed, e.g., clientAuth in extendedKeyUsage
  # For example: -extfile openssl_client.cnf -extensions v3_client

# Clean up CSR files and serial files from the generation directory
rm "$CONFIG_DIR"/*.csr
rm "$CONFIG_DIR"/*.srl 2>/dev/null || true


echo "Certificates generated successfully in $CONFIG_DIR."

if [ -d "$OUTPUT_DIR" ]; then
  echo "Copying certificates to $OUTPUT_DIR..."
  if [ "$FORCE_REGENERATE" = "true" ]; then
    rm -f "$OUTPUT_DIR"/*.*
  fi
  cp "$CONFIG_DIR/ca.crt" "$CONFIG_DIR/ca.key" \
     "$CONFIG_DIR/client.crt" "$CONFIG_DIR/client.key" \
     "$CONFIG_DIR/server.crt" "$CONFIG_DIR/server.key" \
     "$OUTPUT_DIR/"
  echo "Certificates copied to $OUTPUT_DIR."
else
  echo "Warning: $OUTPUT_DIR directory not found. Certificates remain in the container's $CONFIG_DIR directory."
fi

echo "Certificate generation complete."
