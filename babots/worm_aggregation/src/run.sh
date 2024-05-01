#!/bin/bash

# Set paths
EXE_PATH="../../../build/bin/Release/worm_aggregation"
JSON_PATH="parameters.json"
LOG_FOLDER="logs"

# Check if log file name is provided as argument, otherwise use default
if [ $# -eq 0 ]; then
    echo "Usage: $0 <log_filename>"
    exit 1
fi

LOG_FILENAME="$1"


# Launch the executable
"$EXE_PATH" -s 1000 -i "$JSON_PATH"

# Check if log.json exists
if [ -f "log.json" ]; then
    # Create logs folder if it doesn't exist
    mkdir -p "$LOG_FOLDER"

    # Move log.json to the logs folder with the specified name
    mv "log.json" "$LOG_FOLDER"/"$LOG_FILENAME"
else
    echo "Error: $LOG_FILENAME not found."
fi