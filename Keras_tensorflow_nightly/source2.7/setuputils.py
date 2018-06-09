import codecs
import os.path
import re


def read(*parts):
    file_path = os.path.join(os.path.dirname(__file__), *parts)
    with codecs.open(file_path, 'r') as fobj:
        content = fobj.read()
    return content


def find_version(*parts):
    version_file = read(*parts)
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]',
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')
