from setuptools import setup, find_packages

setup(
    name="t2pmhc",
    author="Mark Polster",
    version="1.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "t2pmhc": ["data/**/*",
                   "utils/*.json",]
    },
    entry_points={
        "console_scripts": [
            "t2pmhc = t2pmhc.__main__:run_t2pmhc",
        ]
    },
)