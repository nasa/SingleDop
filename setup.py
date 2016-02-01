from setuptools import setup

setup(
    name='SingleDop',
    version='0.9',
    author='Timothy Lang',
    author_email='timothy.j.lang@nasa.gov',
    packages=['singledop',],
    license='LICENSE.md',
    description='Single Doppler Retrieval Toolkit',
    long_description=open('description.txt').read(),
    install_requires=['arm_pyart', 'pytda'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console"
        ],
)
