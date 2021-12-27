import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="min2net",
    version="1.0.0",
    author="INTERFACES",
    author_email="IoBT.VISTEC@gmail.com",
    description="MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://MIN2Net.github.io",
    download_url="https://github.com/IoBT-VISTEC/MIN2Net/releases",
    project_urls={
        "Bug Tracker": "https://github.com/IoBT-VISTEC/MIN2Net/issues",
        "Documentation": "https://MIN2Net.github.io",
        "Source Code": "https://github.com/IoBT-VISTEC/MIN2Net",
    },
    license="Apache Software License",
    keywords=[
        "Brain-computer Interfaces"
        "BCI", 
        "Motor Imagery",
        "MI", 
        "Multi-task Learning",
        "Deep Metric Learning",
        "DML", 
        "Autoencoder",
        "AE",
        "EEG Classifier"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "setuptools>=42",
        "wheel>=0.37.0",
        # "tensorflow-gpu==2.2.0", not support tensorflow-gpu via pip
        "tensorflow-addons==0.9.1",
        "scikit-learn>=0.24.1",
        "wget>=3.2"
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',

)
