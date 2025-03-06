from zenml.client import Client

def fetch_best_model_outputs(pipeline_name: str, step_name: str):
    """
    Fetch the best model and evaluation metrics from a ZenML pipeline's last successful run.

    Args:
        pipeline_name (str): Name of the pipeline (e.g., "regression_pipeline").
        step_name (str): Name of the step that selects the best model (e.g., "select_best_regression_model").

    Returns:
        tuple: (best_model, metrics) if found, otherwise (None, None).
    """
    try:
        # Fetch the pipeline object
        pipeline = Client().get_pipeline(pipeline_name)
        if not pipeline:
            print(f"Pipeline '{pipeline_name}' not found.")
            return None, None

        # Get the last successful pipeline run
        pipeline_run = pipeline.last_successful_run
        if not pipeline_run:
            print(f"No successful run found for pipeline: {pipeline_name}")
            return None, None

        # Retrieve step outputs
        step = pipeline_run.steps.get(step_name)
        if not step:
            print(f"Step '{step_name}' not found in pipeline run.")
            return None, None

        outputs = step.outputs

        # Ensure the necessary outputs exist before loading
        best_model = outputs.get("best_model")
        metrics = outputs.get("metrics")

        if best_model is None or metrics is None:
            print(f"Missing expected outputs in step '{step_name}'.")
            return None, None

        return best_model.load(), metrics.load()

    except Exception as e:
        print(f"Error fetching outputs for {pipeline_name}: {str(e)}")
        return None, None
