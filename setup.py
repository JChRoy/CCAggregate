from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ccagg",
    version="0.1.0",
    description="Resampling-aggregated multiview CCA with stability "
                "selection and permutation inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jean-Charles Roy",
    url="https://github.com/JChRoy/CCAggregate",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3", 
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.2.2, <1.6",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "joblib>=1.1",
        "tqdm>=4.60",
        # v3.0.0 renamed weights_ -> weights and changed the model API.
        # ccagg relies on weights_, loadings_(), pairwise_correlations()
        # and transform() returning one array per view.
        "cca-zoo>=2.3,<3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
