# Copyright 2021 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
"""KFP Component to calculate evaluation metric for AutoML model and take decision"""

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
    location: str,  # "us-central1",
    experiment_name: str,
    thresholds_dict_str: str,
    model: Input[Artifact],
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("dep_decision", str)]):  # Return parameter.

    import json
    import logging
    from google.cloud import aiplatform
    
    #Convert Threshold JSON to DICT
    thresholds_dict = json.loads(thresholds_dict_str)
    rmse_threshold = thresholds_dict['rmse_threshold']
    
    #Get eval metrics stored by training job, from Vertex MLMD
    aiplatform.init(
        project=project,
        location=location
    )

    #Read metrics from MLMD
    eval_metrics = aiplatform.get_experiment_df(experiment_name).to_dict(orient="r")
    eval_rmse = eval_metrics[-1]['metric.rmse']
    
    #Log metrics to Vertex
    metrics.log_metric("eval_rmse", eval_rmse)
    
    #Check if model metric is better than threshold metric
    if eval_rmse <= rmse_threshold:
        dep_decision = "true"
    else:
        dep_decision = "false"
    logging.info("deployment decision is %s", dep_decision)

    return (dep_decision,)

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
