from setuptools import setup

setup(
    name="recomiendame",
    version='0.1',
    py_modules=['cli'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        recomiendame=cli:main
    ''',
)