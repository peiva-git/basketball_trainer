from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

dependencies = [
    'paddlepaddle==2.5.1',
    'paddleseg==2.8.*',
    'Pillow>=10.0.*',
    'numpy>=1.25.*',
    'opencv-python>=4.5.*',
    'pdoc==14.1.*'
]

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
    install_requires=dependencies,
    dependency_links=[
        'https://mirror.baidu.com/pypi/simple'
    ],
    entry_points={
        'console_scripts': [
            'train = basketballtrainer.cli:train_model_command'
        ]
    },
)
