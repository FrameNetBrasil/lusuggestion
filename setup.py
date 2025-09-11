#!/usr/bin/env python3
"""
Setup configuration for FrameNet LU Generator

This package provides tools for generating Brazilian Portuguese lexical units
for FrameNet frames using large language models.
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "FrameNet Lexical Unit Generator - A tool for generating Brazilian Portuguese lexical units for FrameNet frames using large language models."

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove inline comments and version constraints for basic packages
                    package = line.split("#")[0].strip()
                    if package and not package.startswith(("-", "logging", "os", "sys", "time", "argparse", "pathlib")):
                        requirements.append(package)
    return requirements

setup(
    name="framenet-lu-generator",
    version="1.0.0",
    author="FrameNet Brasil Team",
    author_email="framenetbr@gmail.com",
    description="Brazilian Portuguese lexical unit generator for FrameNet frames using LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/FrameNetBrasil/lusuggestion",
    packages=find_packages(),
    py_modules=["framenet_generator", "app"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "gpu": [
            "llama-cpp-python[cublas]",
        ],
    },
    entry_points={
        "console_scripts": [
            "framenet-lu-api=app:main",
            "framenet-lu-cli=framenet_lu_generator_llama:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords="framenet nlp linguistics portuguese llm ai language-model lexical-units",
    project_urls={
        "Bug Reports": "https://github.com/FrameNetBrasil/lusuggestion/issues",
        "Source": "https://github.com/FrameNetBrasil/lusuggestion",
        "Documentation": "https://github.com/FrameNetBrasil/lusuggestion/blob/main/README.md",
    },
)