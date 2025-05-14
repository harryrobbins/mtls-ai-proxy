# cert-generator/Dockerfile
FROM alpine:latest

# Add openssl
RUN apk add --no-cache openssl

WORKDIR /certs

# Copy the script and the new OpenSSL CA config file
COPY generate_certs.sh .
COPY openssl_ca.cnf .
COPY openssl_server.cnf .

RUN chmod +x generate_certs.sh

CMD ["./generate_certs.sh"]
