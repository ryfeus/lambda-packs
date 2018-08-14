# $Id: HDF.py,v 1.3 2005-07-14 01:36:41 gosselin_a Exp $
# $Log: not supported by cvs2svn $
# Revision 1.2  2004/08/02 15:36:04  gosselin
# pyhdf-0.7-1
#
# Revision 1.1  2004/08/02 15:22:59  gosselin
# Initial revision
#
# Author: Andre Gosselin
#         Maurice-Lamontagne Institute
#         gosselina@dfo-mpo.gc.ca

"""
Basic API (:mod:`pyhdf.HDF`)
============================

A module of the pyhdf package implementing the basic API of the
NCSA HDF4 library.
(see: hdf.ncsa.uiuc.edu)

Introduction
------------
The role of the HDF module is to provide support to other modules of the
pyhdf package. It defines constants specifying file opening modes and
various data types, methods for accessing files, plus a few utility
functions to query library version and check if a file is an HDF one.

It should be noted that, among the modules of the pyhdf package, SD is
special in the sense that it is self-contained and does not need support
from the HDF module. For example, SD provides its own file opening and
closing methods, whereas VS uses methods of the HDF.HDF class for that.

Functions and classes summary
-----------------------------
The HDF module provides the following classes.

  HC
      The HC class holds constants defining opening modes and
      various data types.

  HDF
      The HDF class provides methods to open and close an HDF file,
      and return instances of the major HDF APIs (except SD).

      To instantiate an HDF class, call the HDF() constructor.

      methods:
        constructors:
          HDF()     open an HDF file, creating the file if necessary,
                    and return an HDF instance
          vstart()  initialize the VS (Vdata) API over the HDF file and
                    return a VS instance
          vgstart() initialize the V (Vgroup) interface over the HDF file
                    and return a V instance.


        closing file
          close()  close the HDF file

        inquiry
          getfileversion()  return info about the version of the  HDF file

The HDF module also offers the following functions.

  inquiry
    getlibversion()    return info about the version of the library
    ishdf()            determine whether a file is an HDF file


"""

import os, sys, types

from . import hdfext as _C
from .six.moves import xrange
from .HC import HC

# NOTE: The vstart() and vgstart() modules need to access the
#       VS and V modules, resp. We could simply import those
#       two modules, but then they would always be loaded and this
#       may not be what the user wants. Instead of forcing the
#       systematic import, we import the package `pyhdf',
#       and access the needed constructors by writing
#       'pyhdf.VS.VS()' and 'pyhdf.V.V()'. Until the VS or
#       V modules are imported, those statements will give an
#       error (undefined attribute). Once the user has imported
#       the modules, the error will disappear.

import pyhdf

from .error import HDF4Error, _checkErr

# List of names we want to be imported by an "from pyhdf.HDF import *"
# statement

__all__ = ['HDF', 'HDF4Error',
           'HC',
           'getlibversion', 'ishdf']

def getlibversion():
    """Get the library version info.

    Args:
      no argument
    Returns:
      4-element tuple with the following components:
        -major version number (int)
        -minor version number (int)
        -complete library version number (int)
        -additional information (string)

    C library equivalent : Hgetlibversion
                                                   """

    status, major_v, minor_v, release, info = _C.Hgetlibversion()
    _checkErr('getlibversion', status, "cannot get lib version")
    return major_v, minor_v, release, info

def ishdf(filename):
    """Determine whether a file is an HDF file.

    Args:
      filename  name of the file to check
    Returns:
      1 if the file is an HDF file, 0 otherwise

    C library equivalent : Hishdf
                                            """

    return _C.Hishdf(filename)


class HDF(object):
    """The HDF class encapsulates the basic HDF functions.
    Its main use is to open and close an HDF file, and return
    instances of the major HDF APIs (except for SD).
    To instantiate an HDF class, call the HDF() constructor. """

    def __init__(self, path, mode=HC.READ, nblocks=0):
        """HDF constructor: open an HDF file, creating the file if
        necessary.

        Args:
          path    name of the HDF file to open
          mode    file opening mode; this mode is a set of binary flags
                  which can be ored together

                      HC.CREATE   combined with HC.WRITE to create file
                                  if it does not exist
                      HC.READ     open file in read-only access (default)
                      HC.TRUNC    if combined with HC.WRITE, overwrite
                                  file if it already exists
                      HC.WRITE    open file in read-write mode; if file
                                  exists it is updated, unless HC.TRUNC is
                                  set, in which case it is erased and
                                  recreated; if file does not exist, an
                                  error is raised unless HC.CREATE is set,
                                  in which case the file is created

                   Note an important difference in the way CREATE is
                   handled by the HDF C library and the pyhdf package.
                   In the C library, CREATE indicates that a new file should
                   always be created, overwriting an existing one if
                   any. For pyhdf, CREATE indicates a new file should be
                   created only if it does not exist, and the overwriting
                   of an already existing file must be explicitly asked
                   for by setting the TRUNC flag.

                   Those differences were introduced so as to harmonize
                   the way files are opened in the pycdf and pyhdf
                   packages. Also, this solves a limitation in the
                   hdf (and netCDF) library, where there is no easy way
                   to implement the frequent requirement that an existent
                   file be opened in read-write mode, or created
                   if it does not exist.

          nblocks  number of data descriptor blocks in a block wit which
                   to create the file; the parameter is ignored if the file
                   is not created; 0 asks to use the default

        Returns:
          an HDF instance

        C library equivalent : Hopen
                                                     """
        # Private attributes:
        #  _id:       file id (NOTE: not compatile with the SD file id)

        # See if file exists.
        exists = os.path.exists(path)

        if HC.WRITE & mode:
            if exists:
                if HC.TRUNC & mode:
                    try:
                        os.remove(path)
                    except Exception as msg:
                        raise HDF4Error(msg)
                    mode = HC.CREATE
                else:
                    mode = HC.WRITE
            else:
                if HC.CREATE & mode:
                    mode = HC.CREATE
                else:
                    raise HDF4Error("HDF: no such file")
        else:
            if exists:
                if mode & HC.READ:
                    mode = HC.READ     # clean mode
                else:
                    raise HDF4Error("HDF: invalid mode")
            else:
                raise HDF4Error("HDF: no such file")

        id = _C.Hopen(path, mode, nblocks)
        _checkErr('HDF', id, "cannot open %s" % path)
        self._id = id


    def __del__(self):
        """Delete the instance, first calling the end() method
        if not already done.          """

        try:
            if self._id:
                self.close()
        except:
            pass

    def close(self):
        """Close the HDF file.

        Args:
          no argument
        Returns:
          None

        C library equivalent : Hclose
                                                """

        _checkErr('close', _C.Hclose(self._id), "cannot close file")
        self._id = None

    def getfileversion(self):
        """Get file version info.

        Args:
          no argument
        Returns:
          4-element tuple with the following components:
            -major version number (int)
            -minor version number (int)
            -complete library version number (int)
            -additional information (string)

        C library equivalent : Hgetlibversion
                                                   """

        status, major_v, minor_v, release, info = _C.Hgetfileversion(self._id)
        _checkErr('getfileversion', status, "cannot get file version")
        return major_v, minor_v, release, info

    def vstart(self):
        """Initialize the VS API over the file and return a VS instance.

        Args:
          no argument
        Returns:
          VS instance

        C library equivalent : Vstart (in fact: Vinitialize)
                                                              """
        # See note at top of file.
        return pyhdf.VS.VS(self)

    def vgstart(self):
        """Initialize the V API over the file and return a V instance.

        Args:
          no argument
        Returns:
          V instance

        C library equivalent : Vstart (in fact: Vinitialize)
                                                              """
        # See note at top of file.
        return pyhdf.V.V(self)



###########################
# Support functions
###########################


def _array_to_ret(buf, nValues):

    # Convert array 'buf' to a scalar or a list.

    if nValues == 1:
        ret = buf[0]
    else:
        ret = []
        for i in xrange(nValues):
            ret.append(buf[i])
    return ret

def _array_to_str(buf, nValues):

    # Convert array of bytes 'buf' to a string.

    # Return empty string if there is no value.
    if nValues == 0:
        return ""
    # When there is just one value, _array_to_ret returns a scalar
    # over which we cannot iterate.
    if nValues == 1:
        chrs = [chr(buf[0])]
    else:
        chrs = [chr(b) for b in _array_to_ret(buf, nValues)]
    # Strip NULL at end
    if chrs[-1] == '\0':
        del chrs[-1]
    return ''.join(chrs)
