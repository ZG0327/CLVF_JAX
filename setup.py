from pathlib import Path

import setuptools

_CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = "hj_reachability"


def _get_version() -> str:
    init_file = _CURRENT_DIR / PACKAGE_DIR / "__init__.py"
    for line in init_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__") and "=" in line:
            version = line.split("=", 1)[1].strip().strip("'\"")
            if version:
                return version
    raise ValueError(f"`__version__` not defined in `{PACKAGE_DIR}/__init__.py`")


def _read_text(file_name: str) -> str:
    return (_CURRENT_DIR / file_name).read_text(encoding="utf-8")


def _parse_requirements(file_name: str) -> list[str]:
    return [
        line.strip()
        for line in _read_text(file_name).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setuptools.setup(
    name="clvf-jax",
    version=_get_version(),
    description="JAX-based solvers and examples for Control Lyapunov-Value Functions (CLVFs) via Hamilton-Jacobi reachability.",
    long_description=_read_text("README.md"),
    long_description_content_type="text/markdown",
    author="ZG0327",
    url="https://github.com/ZG0327/CLVF_JAX",
    project_urls={
        "Source": "https://github.com/ZG0327/CLVF_JAX",
        "Issues": "https://github.com/ZG0327/CLVF_JAX/issues",
        "Paper": "https://ieeexplore.ieee.org/document/9983827",
    },
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=_parse_requirements("requirements.txt"),
    tests_require=_parse_requirements("requirements-test.txt"),
    python_requires=">=3.8",
    keywords=[
        "control",
        "lyapunov",
        "hamilton-jacobi",
        "reachability",
        "jax",
        "optimal-control",
        "viscosity-solutions",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
