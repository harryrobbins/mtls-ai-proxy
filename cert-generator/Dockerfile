FROM alpine/openssl:latest

WORKDIR /certs

COPY generate_certs.sh .
RUN chmod +x generate_certs.sh

# Default command can be to show usage or run the script
CMD ["./generate_certs.sh"]