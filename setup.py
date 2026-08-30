from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """This function will return the list of requirements"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="black-friday-sales",
    version="0.1",
    description="Black Friday Sales Prediction",
    author="Dhruv Kandwal",
    author_email="dhruv.10040@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)