#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('HISTORY.rst') as history_file:
    history = history_file.read()

def read_readme():
    with open('README.md') as f:
        return f.read()

requirements = [
    'numpy',
    'torch-scope',
    'torch'
]

setup(
    name='LightNER',
    version='0.3.0',
    description='A Toolkit for Pre-trained Sequence Labeling Models Inference',
    long_description= read_readme(),
    author='Lucas Liu',
    author_email='llychinalz@gmail.com',
    url='https://github.com/LiyuanLucasLiu/LightNER',
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=requirements,
    license='Apache License 2.0',
    entry_points={
        'console_scripts': ['lightner=lightner.commands.main:main'],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*