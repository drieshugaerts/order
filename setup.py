from setuptools import setup, find_packages

setup(
    name='order',
    version='0.1',
    description='Software package for order deduction for filling in spreadsheets',
    url='https://github.com/drieshugaerts/order',
    author='Dries Hugaerts',
    author_email='dries.hugaerts@telenet.be',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['autocomplete', 'csv'],
    dependency_links=['git+ssh://git@github.com/samuelkolb/autocomplete.git#user_sim_2']
)
