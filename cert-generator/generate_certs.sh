#!/bin/sh
set -e

# Check if certificates already exist to avoid overwriting unless forced
if [ -f "/certs_output/ca.crt" ] && [ "$FORCE_REGENERATE" != "true" ]; then
  echo "Certificates already exist in /certs_output. Skipping generation."
  echo "Set FORCE_REGENERATE=true to overwrite."
  exit 0
fi

echo "Generating new certificates..."

# 1. Generate CA private key
openssl genrsa -out ca.key 2048

# 2. Generate CA certificate (self-signed)
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt \
  -subj "/C=XX/ST=Testland/L=TestCity/O=LocalTestOrg/OU=DevOps/CN=LocalTestCA"

# 3. Generate Server private key
openssl genrsa -out server.key 2048

# 4. Generate Server Certificate Signing Request (CSR)
openssl req -new -key server.key -out server.csr \
  -subj "/C=XX/ST=Testland/L=TestCity/O=LocalTestOrg/OU=Server/CN=localhost"

# 5. Sign the Server CSR with the CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 3600 -sha256

# 6. Generate Client private key
openssl genrsa -out client.key 2048

# 7. Generate Client Certificate Signing Request (CSR)
openssl req -new -key client.key -out client.csr \
  -subj "/C=XX/ST=Testland/L=TestCity/O=LocalTestOrg/OU=Client/CN=TestClient"

# 8. Sign the Client CSR with the CA
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out client.crt -days 3600 -sha256

# Clean up CSR files and serial files
rm *.csr
rm *.srl

echo "Certificates generated successfully in the current directory (/certs)."
echo "They will be copied to the ./certs directory on your host if volume is mounted."

# If output directory is specified and mounted, copy there
if [ -d "/certs_output" ]; then
  echo "Copying certificates to /certs_output..."
  cp ca.crt ca.key client.crt client.key server.crt server.key /certs_output/
  echo "Certificates copied."
else
  echo "Warning: /certs_output directory not found. Certificates remain in the container's /certs directory."
fi