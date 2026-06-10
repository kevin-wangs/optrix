from setuptools import setup, find_packages

setup(
    name="optrix",
    version="0.2.1",
    description="High-performance parallel compute primitives for heterogeneous hardware",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kevin Wangs",
    url="https://github.com/kevin-wangs/optrix",
    project_urls={
        "Docs": "https://github.com/kevin-wangs/optrix/tree/main/docs",
        "Issues": "https://github.com/kevin-wangs/optrix/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    extras_require={"dev": ["pytest>=7", "pytest-cov", "ruff", "mypy"]},
    entry_points={"console_scripts": ["optrix=optrix.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Hardware",
    ],
)
