#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status.

LOG_FILE="/tmp/ollama_serve.log"

echo "Ollama entrypoint script started."
echo "Logging ollama serve output to $LOG_FILE"

# Start ollama serve in the background, redirecting its stdout and stderr
ollama serve > "$LOG_FILE" 2>&1 &
pid=$! # Get the process ID of ollama serve

# Function to print last few lines of log and kill ollama
fail_and_exit() {
  echo "Ollama failed. Last lines from $LOG_FILE:"
  tail -n 30 "$LOG_FILE" # Print last 30 lines for debugging
  kill $pid >/dev/null 2>&1 || true # Try to kill the background ollama serve process
  echo "Exiting."
  exit 1
}

# Wait for Ollama server to be responsive using "ollama ps"
echo "Waiting for Ollama server to become responsive..."
count=0
max_count=60 # Wait for max 60 seconds
until ollama ps >/dev/null 2>&1; do
  # Check if the background process is still alive
  if ! ps -p $pid > /dev/null; then
    echo "Ollama serve process (PID: $pid) died unexpectedly."
    fail_and_exit
  fi
  sleep 1
  count=$((count+1))
  if [ "$count" -ge "$max_count" ]; then
    echo "Ollama server failed to become responsive (ollama ps) within $max_count seconds."
    fail_and_exit
  fi
  printf "."
done
echo "" # Newline after dots
echo "Ollama server is responsive."
echo "--- Current content of $LOG_FILE (first 20 lines): ---"
head -n 20 "$LOG_FILE"
echo "-------------------------------------------------------"


# Pull the specified model
MODEL_TO_PULL=${OLLAMA_PULL_MODEL:-qwen3:0.6b}
echo "Pulling model: $MODEL_TO_PULL ..."
if ! ollama pull "$MODEL_TO_PULL"; then
    echo "Failed to pull model $MODEL_TO_PULL."
    echo "--- Content of $LOG_FILE on pull failure: ---"
    cat "$LOG_FILE"
    echo "----------------------------------------------"
    kill $pid >/dev/null 2>&1 || true
    exit 1
fi
echo "Model $MODEL_TO_PULL pulled successfully or already exists."

echo "Ollama is running with model $MODEL_TO_PULL. PID: $pid"
wait $pid

echo "Ollama serve process (PID: $pid) has exited."
echo "--- Final content of $LOG_FILE: ---"
cat "$LOG_FILE"
echo "----------------------------------"
exit 0