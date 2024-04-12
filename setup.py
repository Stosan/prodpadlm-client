import os
from pathlib import Path
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def read_readme() -> str:
    """Read the README file."""
    return (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")

requirements = [
    "pydantic-settings",
    "langchain==0.1.14",
    "langchain-core==0.1.40",
    "httpcore==1.0.5",
    "httpx==0.27.0"
]

setup(
    name='prodpadlm_client',
    version='0.1.1.4',
    description='Production LaunchPad for Language Models [Client] - ⚡ Ship Open Source LLMs to production faster and efficiently ⚡',
    author='Ayo Kehinde Samuel',
    packages=find_packages(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    license_files="LICENSE.txt",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.9.0",
    
)