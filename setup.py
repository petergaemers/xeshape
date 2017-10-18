try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#readme = open('README.rst').read()
readme = 'nothing yet'
#history = open('HISTORY.rst').read().replace('.. :changelog:', '')
history = 'no history yet'
requirements = open('requirements.txt').read().splitlines()
test_requirements = requirements

setup(name='xeshape',
      version='0.0.1',
      description='Utilities for extracting LXe waveform shapes',
      long_description=readme + '\n\n' + history,
      author='Jelle Aalbers',
      author_email='j.aalbers@uva.nl',
      url='https://github.com/jelleaalbers/xeshape',
      license='MIT',
      py_modules=['xeshape'],
      install_requires=requirements,
      #keywords='multihist,histogram',
      #test_suite='tests',
      #tests_require=test_requirements,
      classifiers=['Intended Audience :: Developers',
                   'Development Status :: 3 - Alpha',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3'],
      zip_safe=False)
