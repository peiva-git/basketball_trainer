from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required_packages = f.read()

setup(
    name='basketballtrainer',
    version='0.1dev',
    description='Basketball detector training package',
    long_description=readme,
    author='Ivan Pelizon',
    author_email='ivan.pelizon@gmail.com',
    url='https://peiva-git.github.io/basketball_trainer/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'output', 'assets')),
    python_requires='>=3.8,<=3.11',
    install_requires=required_packages,
    entry_points={
        'console_scripts': [
            'train = basketballtrainer.cli:train_model_command'
        ]
    },
)
