#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Shahiryar Saleem",
    author_email='shahiryarsaleem@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="LightRecSys is a lightweight, easy-to-use Python library designed to streamline the process of building recommendation systems. It provides a set of efficient tools for data preprocessing, model training, and prediction. The library supports various types of recommendation algorithms, including collaborative filtering and content-based filtering methods. With a focus on simplicity and performance, LightRecSys is an ideal tool for both beginners and experienced data scientists looking to implement recommendation systems in their projects. Whether you're developing a movie recommendation service, a product recommendation system for an e-commerce platform, or a personalized content recommendation feature for a news website, LightRecSys can help you get the job done quickly and effectively.",
    entry_points={
        'console_scripts': [
            'lightrecsys=lightrecsys.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lightrecsys',
    name='lightrecsys',
    packages=find_packages(include=['lightrecsys', 'lightrecsys.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/shahiryar/lightrecsys',
    version='0.1.0',
    zip_safe=False,
)
