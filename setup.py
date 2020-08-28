from setuptools import setup

setup(name='mridle',
      version='0.0.1',
      description='',
      url='https://github.com/uzh-dqbm-cmi/mridle',
      packages=['mridle'],
      python_requires='>=3.7.3',
      install_requires=[
            'altair',
            'altair_saver',
            'flake8',
            'matplotlib>=3.1.0',
            'numpy>=1.15.0',
            'pandas>=1.0.0',
            'pgeocode',
            'scikit-learn',
            'seaborn'
      ],
      extras_requires={
            'dev': ['jupyter', 'pytest'],
      },
      zip_safe=False)
