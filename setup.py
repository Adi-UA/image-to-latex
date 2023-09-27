# setup.py
# Setup installation for the application

from pathlib import Path

from setuptools import setup


BASE_DIR = Path(__file__).parent


# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]



setup(
    name="image-to-latex-adi",
    version="0.1",
    license="MIT",
    description="Convert images to latex code.",
    author="King Yiu Suen (Updated by Adi)",
    author_email="abanerjee@arizona.edu",
    url="https://github.com/Adi-UA/image-to-latex",
    keywords=[
        "machine-learning",
        "deep-learning",
        "artificial-intelligence",
        "latex",
        "neural-network",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["conf", "figures", "image_to_latex"],
    python_requires=">=3.10",
    install_requires=[required_packages],
)
