
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['absl-py','pandas','google-cloud-aiplatform',
                     'google-cloud','google-cloud-storage','google-cloud-firestore','google-api-python-client', 'google-auth']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Hasan - Vertex AI Taxi Trainer Job'
)
