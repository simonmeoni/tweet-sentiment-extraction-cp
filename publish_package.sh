rm -r dist/
bumpversion --config-file .bumpversion.cfg  --allow-dirty minor setup.py
python setup.py sdist bdist_wheel
python -m twine upload --repository pypi dist/*