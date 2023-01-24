# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from setuptools import setup, find_packages

setup(
    name = 'airbornemetrics', 
    description='A PrimeAir Challenge Metrics',
    version='1.0', 
    license='Apache-2.0',
    install_requires=['pandas>=1.0.1',
                      'numpy',                     
                      ],    
    packages=find_packages(), #include/exclude arguments take * as wildcard, . for any sub-package names
)
