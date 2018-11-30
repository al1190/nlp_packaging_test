from setuptools import find_packages, setup
setup(
    name='nlp_packaging',
    version='0.1.0',
    description='An example package. Generated with https://github.com/ionelmc/cookiecutter-pylibrary',
    author='Annalaura Ianiro',
    author_email='aleteia1190@gmail.com',
    url='TODO',
    install_requires=[
        'spacy',
        'pandas',
        'matplotlib',
        'seaborn',
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    #entry_points={
    #    'console_scripts': [
    #        'nameless = nameless.cli:main',
    #    ]
    #},
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
)
