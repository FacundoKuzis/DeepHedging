import os
from distutils.core import setup

current_directory = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements_list = f.read()

setup(
    name="DeepHedging",
    version="0.1",
    license_file="None.",
    description="Deep Hedging",
    author="Facundo Kuzis",
    author_email="fkuzis@udesa.edu.ar",
    keywords=["Deep Hedging", "reinforcement learning", "machine learning"],
    install_requires=[requirements_list],
    classifiers=[],
)
