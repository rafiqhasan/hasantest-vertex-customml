###Code to Build and Deploy the full pipeline
###This will be used in Cloud Builder

#Initialize pipeline object
from pipelines.train_pipeline import pipeline_controller
import time

PROJECT_ID = "hasanrafiq-test-331814"
REGION="us-central1"

BUCKET_NAME=f"gs://gcs-{PROJECT_ID}"
BUCKET_NAME

PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/"
PIPELINE_ROOT

DISPLAY_NAME = 'hasantest-vertex-customml-pipeline{}'.format(str(int(time.time())))
DISPLAY_NAME

pipe = pipeline_controller(template_path="pipeline.json",
                           display_name="customml-taxi-training", 
                           pipeline_root=PIPELINE_ROOT,
                           project_id=PROJECT_ID,
                           region=REGION)

#Build and Compile pipeline
pipe._build_compile_pipeline()

# #Submit Job
pipe._submit_job()