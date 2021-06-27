import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frankenstein_pytorch", 
    version="0.1",
    author="Chia-wei,Hsu",
    author_email="acctouhou@gamil.com",
    description="Frankenstein Optimizer on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Development Status :: 2 - Pre-Alpha  ',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License ',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
          'torch>=1.6.0',
      ],
    python_requires='>=3.6',
)