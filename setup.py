from setuptools import setup

setup(
    name="einindex",
    version="0.0.3",
    author="Jon Malmaud",
    author_email="malmaud@gmail.com",
    packages=["einindex"],
    package_data={"einindex": ["py.typed", "grammar.lark"]},
    url="https://github.com/malmaud/einindex",
    install_requires=["numpy", "lark-parser", "torch", "dataclasses"],
)
