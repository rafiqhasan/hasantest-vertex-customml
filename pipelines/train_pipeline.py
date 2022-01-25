import os

import kfp
import time
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import NamedTuple, Dict

from google.cloud import aiplatform

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
        
    def _submit_job(self):
        """Method to Submit ML Pipeline job"""
        #Next, define the job:
        ml_pipeline_job = aiplatform.PipelineJob(
            display_name=self.display_name,
            template_path=self.template_path,
            pipeline_root=self.pipeline_root,
            parameter_values={"project": self.project_id, "display_name": self.display_name},
            enable_caching=False
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
            thresholds_dict_str: str = '{"rmse_threshold": 4.8}',
        ):
            
            #Load all reusable custom components
            eval_op = kfp.components.load_component('component_specs/regression_eval_model.yaml')
            hpt_op = kfp.components.load_component('component_specs/hpt.yaml')

            #STEP: For non Auto-ML call
            hpt_args = ['--train_file', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/train.csv',
                         '--eval_file', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/eval.csv',
                         '--model_save_location', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/model/',
                         '--hidden_layers','2',
                         '--experiment_name', "{{$.inputs.parameters['display_name']}}-hpt-job" ,  #This is also a way to pass a parameter as input
                         '--mlmd_region', "{{$.inputs.parameters['pipelineparam--gcp_region']}}" ,
                         '--project', "{{$.inputs.parameters['project']}}"
                        ]
            
            #HPT
            hpt_task = hpt_op(
                project=project,
                display_name=display_name,
                service_account='318948681665-compute@developer.gserviceaccount.com',  ##Needed for Vertex MLMD access
                executor_image_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest',
                package_uri="gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/ml_scripts/trainer-0.1.tar.gz",
                python_module="trainer.task",
                hpt_args=hpt_args,
                metric_id="eval_rmse",
                goal=aiplatform.gapic.StudySpec.MetricSpec.GoalType.MINIMIZE,
                location=gcp_region,
                api_endpoint=str(f"{gcp_region}-aiplatform.googleapis.com")
            )

            #Training
            training_args = ['--train_file', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/train.csv',
                             '--eval_file', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/eval.csv',
                             '--model_save_location', 'gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/model/',
                             '--epochs', '10',
                             '--hidden_layers', str(hpt_task.outputs["hidden_layers"]),
                             '--lr', str(hpt_task.outputs["lr"]),
                             '--experiment_name', str(f"{display_name}-train-job"),
                             '--mlmd_region', str(gcp_region) ,
                             '--project', str(project) ,
                            ]
            
            training_op = gcc_aip.CustomPythonPackageTrainingJobRunOp(
                            project=project,
                            display_name=display_name,
                            service_account='318948681665-compute@developer.gserviceaccount.com',  ##Needed for Vertex MLMD access
                            python_package_gcs_uri="gs://gcs-hasanrafiq-test-331814/ml_data/taxi_dataset/ml_scripts/trainer-0.1.tar.gz",
                            staging_bucket='gs://cloud-ai-platform-35f2698c-5046-4c70-857e-14cb44e3950a/ml_staging',
                            python_module_name="trainer.task",
                            container_uri='us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest',
                            model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest',
                            model_display_name='vertex-customml-taxi',
                            replica_count=1,
                            location=gcp_region,
                            machine_type="n1-standard-4",
                            args=training_args
                          )

            #Model Evaluation
            model_eval_task = eval_op(
                                    project,
                                    gcp_region,
                                    str(f"{display_name}-train-job"),
                                    thresholds_dict_str,
                                    training_op.outputs["model"])
                
            with dsl.Condition(
                model_eval_task.outputs["dep_decision"] == "true",
                name="deploy_decision",
            ):

                endpoint_op = gcc_aip.EndpointCreateOp(
                    project=project,
                    location=gcp_region,
                    display_name="vertex-customml-taxi",
                )

                gcc_aip.ModelDeployOp(
                    model=training_op.outputs["model"],
                    endpoint=endpoint_op.outputs["endpoint"],
                    dedicated_resources_min_replica_count=1,
                    dedicated_resources_max_replica_count=1,
                    dedicated_resources_machine_type="n1-standard-4",
                )
            
        return pipeline_fn
