#!/bin/sh
# Optionally register the stack if it's not already registered
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set || echo "Stack already registered."

# Show the active stack for debugging
zenml stack describe mlflow_stack

# Now run the main pipeline
python main_pipeline.py
