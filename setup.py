from setuptools import setup, find_packages

setup(name='mridle',
      version='0.0.1',
      description='',
      url='https://github.com/uzh-dqbm-cmi/mridle',
      packages=find_packages(),
      python_requires='>=3.7.3',
      install_requires=[
            'altair',
            'altair_saver',
            'datatc',
            'flake8',
            'matplotlib>=3.1.0',
            'numpy>=1.15.0',
            'pandas>=1.0.0',
            'pgeocode',
            'scikit-learn',
            'seaborn',
            'hyperopt'
      ],
      extras_requires={
            'dev': ['jupyter', 'pytest'],
      },
      zip_safe=False)
