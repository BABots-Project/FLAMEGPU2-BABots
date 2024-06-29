#!/bin/bash

# Set paths
EXE_PATH="../../../build/bin/Release/worm_aggregation"
JSON_PATH="parameters.json"
LOG_FOLDER="logs"

# Check if log file name and density are provided as arguments, otherwise use default
if [ $# -lt 2 ]; then
    echo "Usage: $0 <log_filename> <density>"
    exit 1
fi

LOG_FILENAME="$1"
DENSITY="$2"

# Launch the executable
"$EXE_PATH" -s 3000 -i "$JSON_PATH"

# Check if log.json exists
if [ -f "log.json" ]; then


    # Create the density folder inside the logs folder
    DENSITY_FOLDER="${LOG_FOLDER}_${DENSITY}"
    mkdir -p "$DENSITY_FOLDER"

    # Move log.json to the density folder with the specified name
    mv "log.json" "$DENSITY_FOLDER"/"$LOG_FILENAME"
else
    echo "Error: log.json not found."
fi
