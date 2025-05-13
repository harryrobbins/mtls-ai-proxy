FROM alpine:latest

# Add openssl
RUN apk add --no-cache openssl

WORKDIR /certs
COPY generate_certs.sh .
RUN chmod +x generate_certs.sh
CMD ["./generate_certs.sh"]