from setuptools import find_packages,setup

# Read requirement.txt
with open('requirements.txt') as file:
    required=file.read().splitlines()

# Read README.md for long description
with open('README.md','r',encoding='utf-8') as file:
    long_description=file.read()

setup(
    name="Audio Clustering with VAE",
    version="0.1.0",
    author="Jannatul Feardous Nafsi",
    author_email="jannatul15-13868@diu.edu.bd",
    description="Audio Clustering with VAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nafsi177/Audio-and-Lyrics-Multi-Modal-Clustering-VAE.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Brac University.",
        "Programing Language :: Python >=3.12.7",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=required,
    extras_require={
        'dev':[
            'pytest>=7.1.1',
            'pytest-cov>=2.12.1',
        ],
    },
)