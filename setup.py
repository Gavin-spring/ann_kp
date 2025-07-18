from setuptools import setup, find_packages

setup(
    name="ANN_KP",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'generate = Scripts.generate_data:main',
            'preprocess = Scripts.preprocess_data:main',
            'train = Scripts.train_model:main',
            'evaluate = Scripts.evaluate_solvers:main',
            'verify-install = Scripts.verify_installation:verify',
        ],
    }
)