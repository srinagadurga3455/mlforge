from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name = "mlforge",
    version          = "1.0.0",
    author           = "srinagadurga3455",
    description      = "End-to-end ML pipeline library — data ingestion to deployment",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/srinagadurga3455/mlforge",
    packages         = find_packages(),
    python_requires  = ">=3.9",

    install_requires = [
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],

    extras_require = {
        "kaggle"     : ["kaggle>=1.5.0"],
        "huggingface": ["datasets>=2.0.0"],
        "xgboost"    : ["xgboost>=1.7.0"],
        "lightgbm"   : ["lightgbm>=3.3.0"],
        "deployment" : ["fastapi>=0.95.0", "uvicorn>=0.20.0"],
        "databases"  : ["psycopg2-binary>=2.9.0", "pymongo>=4.0.0", "sqlalchemy>=2.0.0"],
        "cloud"      : ["boto3>=1.26.0"],
        "streaming"  : ["kafka-python>=2.0.0"],
        "images"     : ["Pillow>=9.0.0"],
        "all": [
            "kaggle>=1.5.0", "datasets>=2.0.0",
            "xgboost>=1.7.0", "lightgbm>=3.3.0",
            "fastapi>=0.95.0", "uvicorn>=0.20.0",
            "psycopg2-binary>=2.9.0", "pymongo>=4.0.0", "sqlalchemy>=2.0.0",
            "boto3>=1.26.0", "kafka-python>=2.0.0", "Pillow>=9.0.0",
        ],
    },

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords = "machine learning pipeline preprocessing training evaluation deployment",
)
