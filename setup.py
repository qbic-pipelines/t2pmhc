from setuptools import setup, find_packages

setup(
    name="t2pmhc",
    author="Mark Polster",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "t2pmhc": ["data/**/*",
                   "utils/*.json",]
    },
    install_requires=[
        # TODO move requirementx.txt here
    ],
    entry_points={
        "console_scripts": [
            "t2pmhc = t2pmhc.__main__:run_t2pmhc",
        ]
    },
)