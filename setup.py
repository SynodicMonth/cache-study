from setuptools import setup, find_packages

setup(
    name='nkache',
    version='0.1.0',
    packages=find_packages(),
    author='Juni May',
    author_email='juni_may@outlook.com',
    # entry
    entry_points={
        'console_scripts': [
            'nkache = nkache.train:main',
        ]
    },
)
