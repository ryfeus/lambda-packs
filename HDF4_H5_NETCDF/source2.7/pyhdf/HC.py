# $Id: HC.py,v 1.2 2005-07-14 01:36:41 gosselin_a Exp $
# $Log: not supported by cvs2svn $
# Revision 1.1  2004/08/02 15:36:04  gosselin
# Initial revision
#

from . import hdfext as _C

class HC(object):
    """The HC class holds contants defining opening modes and data types.

File opening modes (flags ORed together)

    CREATE       4    create file if it does not exist
    READ         1    read-only mode
    TRUNC      256    truncate if it exists
    WRITE        2    read-write mode

Data types

    CHAR         4    8-bit char
    CHAR8        4    8-bit char
    UCHAR        3    unsigned 8-bit integer (0 to 255)
    UCHAR8       3    unsigned 8-bit integer (0 to 255)
    INT8        20    signed 8-bit integer (-128 to 127)
    UINT8       21    unsigned 8-bit integer (0 to 255)
    INT16       23    signed 16-bit integer
    UINT16      23    unsigned 16-bit integer
    INT32       24    signed 32-bit integer
    UINT32      25    unsigned 32-bit integer
    FLOAT32      5    32-bit floating point
    FLOAT64      6    64-bit floating point

Tags

    DFTAG_NDG  720    dataset
    DFTAG_VH  1962    vdata
    DFTAG_VG  1965    vgroup



    """

    CREATE       = _C.DFACC_CREATE
    READ         = _C.DFACC_READ
    TRUNC        = 0x100          # specific to pyhdf
    WRITE        = _C.DFACC_WRITE

    CHAR         = _C.DFNT_CHAR8
    CHAR8        = _C.DFNT_CHAR8
    UCHAR        = _C.DFNT_UCHAR8
    UCHAR8       = _C.DFNT_UCHAR8
    INT8         = _C.DFNT_INT8
    UINT8        = _C.DFNT_UINT8
    INT16        = _C.DFNT_INT16
    UINT16       = _C.DFNT_UINT16
    INT32        = _C.DFNT_INT32
    UINT32       = _C.DFNT_UINT32
    FLOAT32      = _C.DFNT_FLOAT32
    FLOAT64      = _C.DFNT_FLOAT64

    FULL_INTERLACE = 0
    NO_INTERLACE   =1


# NOTE:
#  INT64 and UINT64 are not yet supported py pyhdf

    DFTAG_NDG = _C.DFTAG_NDG
    DFTAG_VH  = _C.DFTAG_VH
    DFTAG_VG  = _C.DFTAG_VG
