#!/usr/bin/env bash
version="0.1.1"
python setup.py sdist bdist_wheel
sudo twine upload dist/*$version*.whl