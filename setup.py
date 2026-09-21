#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dem",
    version="0.0.1",
    description="Control Variate Score Identity (CVSI) for diffusion samplers",
    url="https://github.com/khaledkah/cvsi",
    license="MIT",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = dem.train:main",
            "eval_command = dem.eval:main",
        ]
    },
)
