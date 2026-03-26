from setuptools import setup, find_packages

setup(
    name="ccagg",
    version="0.1.0",
    description="CCA aggregation tools",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "tqdm",
        "cca-zoo",  
    ],
)