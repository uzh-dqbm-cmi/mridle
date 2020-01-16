from setuptools import setup

setup(name='mridle',
      version='0.0.1',
      description='',
      url='https://github.com/uzh-dqbm-cmi/mridle',
      packages=['mridle'],
      python_requires='>3.5.0',
      install_requires=[
            'numpy>=1.15.0',
            'pandas==0.24.1',
            'matplotlib',
      ],
      extras_requires={
            'dev': ['jupyter'],
      },
      zip_safe=False)
