from setuptools import setup, find_packages

setup(
    name="cyberwave",
    version="0.1.4",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.27.0,<0.28.0",
        "requests>=2.20.0,<3.0.0",
        "aiofiles>=23.2.1,<24.0.0",
        "numpy>=1.26.4,<2.0.0",
        "rerun-sdk>=0.16.0,<0.17.0",
        "pydantic>=2.7.0,<3.0.0",
        "pyyaml>=6.0.1,<7.0.0",
        "jsonschema>=4.21.1,<5.0.0",
        "keyring>=25.0.0,<26.0.0",
    ],
    author="Simone Di Somma",
    author_email="sdisomma@cyberwave.com",
    description="Python SDK for the Cyberwave Digital Twin Platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cyberwave-os/cyberwave-python",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
) 
