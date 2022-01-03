# Copyright 2021 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""KFP Component to run HPT"""

import argparse
import logging
import sys
import json

from google.cloud import aiplatform as aip

from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics
from typing import NamedTuple
from kfp.v2.components import executor

_logger = logging.getLogger(__name__)

#Main body for the function
def main(
    project: str,
    display_name: str,
    service_account: str,
    executor_image_uri: str,
    package_uri: str,
    python_module: str,
    hpt_args: list,
    metric_id: str,
    goal: int,
    metrics: Output[Metrics],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com"
) -> NamedTuple("Outputs", [("lr", float), ("hidden_layers", int)]):
    
    from google.cloud import aiplatform
    import time
    
    _POLLING_INTERVAL_IN_SECONDS = 20

    _JOB_COMPLETE_STATES = (
        'JobState.JOB_STATE_SUCCEEDED',
        'JobState.JOB_STATE_FAILED',
        'JobState.JOB_STATE_CANCELLED',
        'JobState.JOB_STATE_PAUSED'
    )
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)

    # study_spec
    metric = {
        "metric_id": metric_id,
        "goal": goal,
    }

    lr = {
            "parameter_id": "lr",
            "double_value_spec": {"min_value": 0.001, "max_value": 0.05}
    }
    
    hidden_layers = {
            "parameter_id": "hidden_layers",
            "integer_value_spec": {"min_value": 1, "max_value": 4}
    }

    # trial_job_spec
    machine_spec = {
        "machine_type": "n1-standard-4"
    }
    worker_pool_spec = {
        "machine_spec": machine_spec,
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": executor_image_uri,
            "package_uris": [package_uri],
            "python_module": python_module,
            "args": hpt_args,
        },
    }

    #Create HPT Job
    hyperparameter_tuning_job = {
        "display_name": display_name,
        "max_trial_count": 2,
        "parallel_trial_count": 2,
        "study_spec": {
            "metrics": [metric],
            "parameters": [lr, hidden_layers],
            "algorithm": aiplatform.gapic.StudySpec.Algorithm.RANDOM_SEARCH,
        },
        "trial_job_spec": {"worker_pool_specs": [worker_pool_spec], "service_account": service_account},
    }
    parent = f"projects/{project}/locations/{location}"
    name = client.create_hyperparameter_tuning_job(
        parent=parent, hyperparameter_tuning_job=hyperparameter_tuning_job
    )
    print("HPT Job:", name.name)
    
    #Get status of HPT job recursively and wait
    status = client.get_hyperparameter_tuning_job(name=name.name) #initial status
    while str(status.state) not in _JOB_COMPLETE_STATES:
        status = client.get_hyperparameter_tuning_job(name=name.name)
        print("HPT Status:", str(status.state))
        time.sleep(_POLLING_INTERVAL_IN_SECONDS)
        
    #When HPT ends, get best params
    best_high = -9999999999
    best_low = 9999999999
    best_trial = 0
    param_dict = {}
    for trials in status.trials:
        metric_value = trials.final_measurement.metrics[-1].value
        trial_id = trials.id
        if goal == aiplatform.gapic.StudySpec.MetricSpec.GoalType.MINIMIZE and metric_value < best_low:
            best_low = metric_value
            best_trial = trial_id
            best_params = trials.parameters
        elif goal == aiplatform.gapic.StudySpec.MetricSpec.GoalType.MAXIMIZE and metric_value > best_high:
            best_high = metric_value
            best_trial = trial_id
            best_params = trials.parameters
    
    #Log best params to metrics
    for params in best_params:
        param_dict[params.parameter_id] = params.value
        metrics.log_metric(params.parameter_id, params.value)
    
    print("HPT ended")
    return ( param_dict["lr"], int(param_dict["hidden_layers"]))

def executor_main():
    """Main executor."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--executor_input', type=str)
    parser.add_argument('--function_to_execute', type=str)

    args, _ = parser.parse_known_args()
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]

    executor.Executor(
      executor_input=executor_input,
      function_to_execute=function_to_execute).execute()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    executor_main()