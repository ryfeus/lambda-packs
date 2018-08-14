# $Id: error.py,v 1.1 2004-08-02 15:00:34 gosselin Exp $
# $Log: not supported by cvs2svn $

from . import hdfext as _C

# #################
# Error processing
# #################

class HDF4Error(Exception):
    """ An error from inside the HDF4 library.
    """

def _checkErr(procName, val, msg=""):

    if val is None or (not isinstance(val, str) and val < 0):
        #_C._HEprint();
        errCode = _C.HEvalue(1)
        if errCode != 0:
            err = "%s (%d): %s" % (procName, errCode, _C.HEstring(errCode))
        else:
            err = "%s : %s" % (procName, msg)
        raise HDF4Error(err)
