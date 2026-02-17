import os

from setuptools import find_packages, setup


current_directory = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(current_directory, "requirements.txt")

with open(requirements_path, encoding="utf-8") as f:
    requirements_list = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="DeepHedging",
    version="0.1.0",
    description="Deep Hedging",
    author="Facundo Kuzis",
    author_email="fkuzis@udesa.edu.ar",
    keywords=["Deep Hedging", "reinforcement learning", "machine learning"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=requirements_list,
    python_requires=">=3.11,<3.12",
)
