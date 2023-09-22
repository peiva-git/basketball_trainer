from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

dependencies = [
    'paddlepaddle-gpu==2.5.1.post102',
    'pdoc'
]

setup(
    name='basketballtrainer',
    version='0.1dev',
    description='Basketball detector training package',
    long_description=readme,
    author='Ivan Pelizon',
    author_email='ivan.pelizon@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'out', 'assets')),
    python_requires='>=3.8,<=3.11',
    install_requires=dependencies,
    entry_points={
        'console_scripts': [

        ]
    },
)
