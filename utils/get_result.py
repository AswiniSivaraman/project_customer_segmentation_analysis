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
        # Fetch the latest successful pipeline run
        pipeline_run = Client().get_pipeline(pipeline_name).last_successful_run

        if pipeline_run:
            # Retrieve step outputs
            outputs = pipeline_run.get_step(step_name).outputs

            # Load model and metrics
            best_model = outputs["best_model"].load()
            metrics = outputs["metrics"].load()

            return best_model, metrics
        else:
            print(f"No successful run found for pipeline: {pipeline_name}")
            return None, None
    except Exception as e:
        print(f"Error fetching outputs for {pipeline_name}: {str(e)}")
        return None, None
