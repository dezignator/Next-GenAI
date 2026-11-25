"""
Next-GenAI — Package Installation Script
=========================================
Reference: Ali, B., & Chen, G. (2024). Next-generation AI for advanced threat
detection and security enhancement in DNS over HTTPS. Journal of Network and
Computer Applications, Elsevier.

State Key Laboratory of Novel Software Technology
Nanjing University, Nanjing, Jiangsu 210093, China.

Installation:
    pip install -e .                    # Development mode
    pip install .                        # Production mode
    pip install -e ".[tee]"              # With TEE extras
    pip install -e ".[dev]"              # With dev/test extras
    pip install -e ".[all]"              # Everything
"""

import os
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup


# =============================================================================
# GUARDS
# =============================================================================
if sys.version_info < (3, 10):
    raise RuntimeError(
        "Next-GenAI requires Python >= 3.10. "
        "PyTorch 2.1.2 + PyG 2.4.0 + PyFlink 1.18.1 are NOT supported on <3.10."
    )


# =============================================================================
# PATH & METADATA HELPERS
# =============================================================================
ROOT = Path(__file__).parent.resolve()


def _read(rel_path: str) -> str:
    """Read a file relative to repo root."""
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _read_requirements() -> list:
    """Parse requirements.txt, strip comments/blank lines and pip options."""
    reqs = []
    for line in _read("requirements.txt").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", "git+", "http")):
            continue  # skip pip options / git refs
        reqs.append(line)
    return reqs


def _get_version() -> str:
    """
    Version is read from src/__init__.py  (single source of truth).
    Falls back to 0.1.0 if not found.
    """
    init_file = ROOT / "src" / "__init__.py"
    if not init_file.exists():
        return "0.1.0"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        init_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else "0.1.0"


LONG_DESCRIPTION = _read("README.md") if (ROOT / "README.md").exists() else ""


# =============================================================================
# EXTRAS
# =============================================================================
EXTRAS_REQUIRE = {
    # TEE backend dependencies (Intel SGX SDK & Open Enclave SDK must be
    # installed at OS-level — see docs/tee_setup.md)
    "tee": [
        "cryptography==41.0.7",
        "pyopenssl==23.3.0",
        "pyjwt==2.8.0",
        "pyasn1==0.5.1",
    ],
    # Streaming pipeline
    "streaming": [
        "kafka-python==2.0.2",
        "apache-flink==1.18.1",
        "confluent-kafka==2.3.0",
    ],
    # Explainability
    "xai": [
        "shap==0.44.1",
        "lime==0.2.0.1",
        "captum==0.7.0",
        "alibi==0.9.4",
    ],
    # Development & testing
    "dev": [
        "pytest==7.4.3",
        "pytest-cov==4.1.0",
        "pytest-asyncio==0.21.1",
        "pytest-mock==3.12.0",
        "hypothesis==6.92.1",
        "black==23.12.0",
        "isort==5.13.2",
        "flake8==6.1.0",
        "mypy==1.7.1",
        "pre-commit==3.6.0",
    ],
    # Documentation
    "docs": [
        "sphinx==7.2.6",
        "sphinx-rtd-theme==2.0.0",
        "myst-parser==2.0.0",
    ],
}

# "all" = union of every extra
EXTRAS_REQUIRE["all"] = sorted(
    {pkg for group in EXTRAS_REQUIRE.values() for pkg in group}
)


# =============================================================================
# SETUP
# =============================================================================
setup(
    # ---------- Metadata ----------
    name="next-gen-ai",
    version=_get_version(),
    description=(
        "Next-Generation AI for Advanced Threat Detection and Security "
        "Enhancement in DNS over HTTPS (DoH)"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Basharat Ali, Guihai Chen",
    author_email="basharat@smail.nju.edu.cn, gchen@nju.edu.cn",
    maintainer="Next-GenAI Research Team",
    url="https://github.com/<your-username>/Next-GenAI",
    project_urls={
        "Paper": (
            "https://www.sciencedirect.com/journal/"
            "journal-of-network-and-computer-applications"
        ),
        "Bug Tracker": "https://github.com/<your-username>/Next-GenAI/issues",
        "Documentation": "https://github.com/<your-username>/Next-GenAI/tree/main/docs",
        "Source Code": "https://github.com/<your-username>/Next-GenAI",
    },

    # ---------- Licensing ----------
    license="MIT",
    license_files=("LICENSE",),

    # ---------- Classifiers (PyPI) ----------
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
    ],

    # ---------- Packaging ----------
    keywords=[
        "DNS over HTTPS", "DoH", "network security", "capsule networks",
        "graph transformers", "contrastive self-supervised learning",
        "adversarial robustness", "trusted execution environment",
        "SHAP", "LIME", "Kafka", "Flink", "anomaly detection",
    ],
    python_requires=">=3.10,<3.11",
    install_requires=_read_requirements(),
    extras_require=EXTRAS_REQUIRE,

    packages=find_packages(
        where=".",
        include=["src", "src.*"],
        exclude=["tests", "tests.*", "notebooks", "docs", "scripts"],
    ),
    package_dir={"": "."},

    # ---------- Data files (configs, YAML, config schemas) ----------
    package_data={
        "src": ["py.typed"],
    },
    include_package_data=True,

    # ---------- Entry points (CLI) ----------
    entry_points={
        "console_scripts": [
            # Training / evaluation
            "nextgenai-train=scripts.train:main",
            "nextgenai-evaluate=scripts.evaluate:main",
            "nextgenai-attacks=scripts.simulate_attacks:main",
            # Streaming pipeline
            "nextgenai-pipeline=scripts.run_pipeline:main",
            # Benchmarking
            "nextgenai-benchmark=scripts.benchmark:main",
        ],
    },

    # ---------- Misc ----------
    zip_safe=False,
    platforms=["Linux"],
)
