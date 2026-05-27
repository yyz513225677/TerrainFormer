from setuptools import setup, find_packages

setup(
    name="terrainformer",
    version="0.1.0",
    description="Autonomous Off-Road Navigation with World Models and Decision Transformers",
    author="Leonardo Yang",
    author_email="",
    url="https://github.com/your-org/terrainformer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "open3d>=0.17.0",
            "plotly>=5.14.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
