import os

import kfp
import time
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import NamedTuple

#Main pipeline class
class pipeline_controller():
    def __init__(self, template_path, display_name, pipeline_root, project_id, region):
        self.template_path = template_path
        self.display_name = display_name
        self.pipeline_root = pipeline_root
        self.project_id = project_id
        self.region = region
    
    def _build_compile_pipeline(self):
        """Method to build and compile pipeline"""
        self.pipeline = self._get_pipeline()
        compiler.Compiler().compile(
            pipeline_func=self.pipeline, package_path=self.template_path
        )
        
        ##Write JSON file to GCS
        
    def _submit_job(self):
        """Method to Submit ML Pipeline job"""
        #Next, define the job:
        ml_pipeline_job = aiplatform.PipelineJob(
            display_name=self.display_name,
            template_path=self.template_path,
            pipeline_root=self.pipeline_root,
            parameter_values={"project": self.project_id, "display_name": self.display_name},
            enable_caching=True
        )

        #And finally, run the job:
        ml_pipeline_job.submit()
    
    def _get_pipeline(self):
        """Main method to Create pipeline"""
        @pipeline(name=self.display_name,
                          pipeline_root=self.pipeline_root)
        def pipeline_fn(
            display_name: str = self.display_name,
            project: str = self.project_id,
            gcp_region: str = self.region,
            api_endpoint: str = "us-central1-aiplatform.googleapis.com",
            thresholds_dict_str: str = '{"auRoc": 0.95}',
        ):
            
            #Load all reusable custom components
            #Option 1 -> Python function component wrapped as reusable function
            # eval_op = kfp.components.load_component('component_specs/classification_eval_model.yaml')
            
            #Option 2 -> Python component wrapped as reusable package( preferred )
            # eval_op = kfp.components.load_component('component_specs/classification_eval_model_v2.yaml')

            #STEP: For non Auto-ML call
            training_args = ['--train_file', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/train.csv',
                             '--eval_file', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/eval.csv',
                             '--model_save_location', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/model/',
                             '--epochs', '10',
                             '--hidden_layers','2'
                            ]

            training_op = gcc_aip.CustomPythonPackageTrainingJobRunOp(
                            project=project,
                            display_name=display_name,
                            python_package_gcs_uri="gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/ml_scripts/trainer-0.1.tar.gz",
                            staging_bucket='gs://cloud-ai-platform-35f2698c-5046-4c70-857e-14cb44e3950a/ml_staging',
                            # base_output_dir='gs://cloud-ai-platform-35f2698c-5046-4c70-857e-14cb44e3950a/ml_staging',
                            python_module_name="trainer.task",
                            container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest',
                            replica_count=1,
                            location=gcp_region,
                            machine_type="n1-standard-4",
                            args=training_args
                          )

#             model_eval_task = eval_op(
#                 project,
#                 gcp_region,
#                 api_endpoint,
#                 thresholds_dict_str,
#                 training_op.outputs["model"],
#             )

#             with dsl.Condition(
#                 model_eval_task.outputs["dep_decision"] == "true",
#                 name="deploy_decision",
#             ):

#                 endpoint_op = gcc_aip.EndpointCreateOp(
#                     project=project,
#                     location=gcp_region,
#                     display_name="train-automl-beans",
#                 )

#                 gcc_aip.ModelDeployOp(
#                     model=training_op.outputs["model"],
#                     endpoint=endpoint_op.outputs["endpoint"],
#                     dedicated_resources_min_replica_count=1,
#                     dedicated_resources_max_replica_count=1,
#                     dedicated_resources_machine_type="n1-standard-4",
#                 )
            
        return pipeline_fn
