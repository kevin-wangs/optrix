from setuptools import setup, find_packages

setup(
    name="optrix",
    version="0.1.0",
    description="High-performance parallel compute primitives for heterogeneous hardware",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kevin Wangs",
    url="https://github.com/kevin-wangs/optrix",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    extras_require={
        "dev": ["pytest>=7", "pytest-cov", "ruff"],
        "rocm": [],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Hardware",
    ],
)
