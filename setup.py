from os.path import join
import logging
import io
from setuptools import setup, find_packages

package_name = "overlap_probe_eprofile"


def version ( ) :

    package_namespace = {}

    try:

        execfile(join(package_name,"version.py"), package_namespace)

    except NameError:

        exec(
            open(join(package_name,"version.py")).read(),
            globals(),
            package_namespace
        )

    return package_namespace["__version__"]

def long_description():

    try:
        with io.open("README.md", encoding="utf-8") as readme:

            return readme.read()

    except IOError:

        logging.warning(
            "No README.md found, package long_description will be empty"
        )

        return None

setup(
    name = package_name,
    version = version(),
    description = "EPROFILE overlap temperature correction code",
    long_description = long_description(),
    url = "https://github.com/martin-obs/OVERLAP_PROBE_EPROFILE",
    packages = find_packages(),
    install_requires=['numpy',
                      'netCDF4',
                      'scipy',
                      'matplotlib',
                      'datetime',
                      'pandas',
                      'matplotlib',
                      'more_itertools',
                      ],
)

