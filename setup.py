import setuptools

packages=setuptools.find_packages()

setuptools.setup(
    name="bispectral-networks",
    version="0.0.1",
    packages=packages,
	include_package_data=True,
	package_data={'': [
		'*.pt',
	]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
