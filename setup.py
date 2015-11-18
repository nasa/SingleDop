from setuptools import setup

setup(
    name='SingleDop',
    version='0.8',
    author='Timothy Lang',
    author_email='timothy.j.lang@nasa.gov',
    packages=['singledop',],
    license='LICENSE.txt',
    description='Single Doppler Retrieval Toolkit',
    long_description=open('description.txt').read(),
    install_requires=['arm_pyart', 'pytda', 'xray'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console"
        ],

)
