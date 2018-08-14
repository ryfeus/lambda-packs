# $Id: SD.py,v 1.10 2008-08-05 00:20:44 gosselin_a Exp $
# $Log: not supported by cvs2svn $
# Revision 1.9  2008/06/30 02:59:57  gosselin_a
# Fixed definition of equivNumericTypes list.
#
# Revision 1.8  2008/06/30 02:41:44  gosselin_a
# Preleminary check-in of changes leading to the 0.8 revision.
#   - switch to numpy, Numeric now unsupported
#   - better documentation of the compression features
#   - some bug fixes
#
# Revision 1.7  2005/07/14 01:36:41  gosselin_a
# pyhdf-0.7-3
# Ported to HDF4.2r1.
# Support for SZIP compression on SDS datasets.
# All classes are now 'new-style' classes, deriving from 'object'.
# Update documentation.
#
# Revision 1.6  2005/01/25 18:17:53  gosselin_a
# Importer le symbole 'HDF4Error' a partir du module SD.
#
# Revision 1.5  2004/08/02 17:06:20  gosselin
# pyhdf-0.7.2
#
# Revision 1.4  2004/08/02 15:36:04  gosselin
# pyhdf-0.7-1
#
# Revision 1.3  2004/08/02 15:22:59  gosselin
# pyhdf -0.6-1
#
# Revision 1.2  2004/08/02 15:00:34  gosselin
# pyhdf 0.5-2
#
# Author: Andre Gosselin
#         Maurice Lamontagne Institute
#         Andre.Gosselin@dfo-mpo.gc.ca

"""
SD (scientific dataset) API (:mod:`pyhdf.SD`)
=============================================

A module of the pyhdf package implementing the SD (scientific
dataset) API of the NCSA HDF4 library.
(see: hdf.ncsa.uiuc.edu)

Introduction
------------
SD is one of the modules composing pyhdf, a python package implementing
the NCSA HDF library and letting one manage HDF files from within a python
program. Two versions of the HDF library currently exist, version 4 and
version 5. pyhdf only implements version 4 of the library. Many
different APIs are to be found inside the HDF4 specification.
Currently, pyhdf implements just a few of those: the SD, VS and V APIs.
Other APIs should be added in the future (GR, AN, etc).

The SD module implements the SD API of the HDF4 library, supporting what
are known as "scientific datasets". The HDF SD API has many similarities
with the netCDF API, another popular API for dealing with scientific
datasets. netCDF files can be in fact read and modified using the SD
module (but cannot be created from scratch).

SD module key features
----------------------
SD key features are as follows.

- Almost every routine of the original SD API has been implemented inside
  pyhdf. Only a few have been ignored, most of them being of a rare use:

  - SDsetnbitdataset()
  - All chunking/tiling routines : SDgetchunkinfo(), SDreadchunk(),
    SDsetchunk(), SDsetchunkcache(), SDwritechunk()
  - SDsetblocksize()
  - SDisdimval_bwcomp(), SDsetdimval_comp()

- It is quite straightforward to go from a C version to a python version
  of a program accessing the SD API, and to learn SD usage by refering to
  the C API documentation.

- A few high-level python methods have been developped to ease
  programmers task. Of greatest interest are those allowing access
  to SD datasets through familiar python idioms.

  - Attributes can be read/written like ordinary python class
    attributes.
  - Datasets can be read/written like ordinary python lists using
    multidimensional indices and so-called "extended slice syntax", with
    strides allowed.

    See "High level attribute access" and "High level variable access"
    sections for details.

  - SD offers methods to retrieve a dictionnary of the attributes,
    dimensions and variables defined on a dataset, and of the attributes
    set on a variable and a dimension. Querying a dataset is thus geatly
    simplified.

- SD datasets are read/written through "numpy", a sophisticated
  python package for efficiently handling multi-dimensional arrays of
  numbers. numpy can nicely extend the SD functionnality, eg.
  adding/subtracting arrays with the '+/-' operators.

Accessing the SD module
-----------------------
To access the SD API a python program can say one of:

  >>> import pyhdf.SD        # must prefix names with "pyhdf.SD."
  >>> from pyhdf import SD   # must prefix names with "SD."
  >>> from pyhdf.SD import * # names need no prefix

This document assumes the last import style is used.

numpy will also need to be imported:

  >>> from numpy import *

Package components
------------------
pyhdf is a proper Python package, eg a collection of modules stored under
a directory whose name is that of the package and which stores an
__init__.py file. Following the normal installation procedure, this
directory will be <python-lib>/site-packages/pyhdf', where <python-lib>
stands for the python installation directory.

For each HDF API exists a corresponding set of modules.

The following modules are related to the SD API.

  _hdfext
    C extension module responsible for wrapping the HDF
    C-library for all python modules
  hdfext
    python module implementing some utility functions
    complementing the _hdfext extension module
  error
    defines the HDF4Error exception
  SD
    python module wrapping the SD API routines inside
    an OOP framework

_hdfext and hdfext were generated using the SWIG preprocessor.
SWIG is however *not* needed to run the package. Those two modules
are meant to do their work in the background, and should never be called
directly. Only 'pyhdf.SD' should be imported by the user program.

Prerequisites
-------------
The following software must be installed in order for pyhdf release 0.8 to
work.

  HDF (v4) library, release 4.2r1
    pyhdf does *not* include the HDF4 library, which must
    be installed separately.

    HDF is available at:
    "http://hdf.ncsa.uiuc.edu/obtain.html".

  HDF4.2r1 in turn relies on the following packages :

    ======= ============== ===========================================
    libjpeg (jpeg library) release 6b
    libz    (zlib library) release 1.1.4 or above
    libsz   (SZIP library) release 2.0; this package is optional
                           if pyhdf is installed with NOSZIP macro set
    ======= ============== ===========================================

The SD module also needs:

  numpy python package
    SD variables are read/written using the array data type provided
    by the python NumPy package. Note that since version 0.8 of
    pyhdf, version 1.0.5 or above of NumPy is needed.

    numpy is available at:
    "http://www.numpy.org".

Documentation
-------------
pyhdf has been written so as to stick as closely as possible to
the naming conventions and calling sequences documented inside the
"HDF User s Guide" manual. Even if pyhdf gives an OOP twist
to the C API, the manual can be easily used as a documentary source
for pyhdf, once the class to which a function belongs has been
identified, and of course once requirements imposed by the Python
langage have been taken into account. Consequently, this documentation
will not attempt to provide an exhaustive coverage of the HDF SD
API. For this, the user is referred to the above manual.
The documentation of each pyhdf method will indicate the name
of the equivalent routine inside the C API.

This document (in both its text and html versions) has been completely
produced using "pydoc", the Python documentation generator (which
made its debut in the 2.1 Python release). pydoc can also be used
as an on-line help tool. For example, to know everything about
the SD.SDS class, say:

  >>> from pydoc import help
  >>> from pyhdf.SD import *
  >>> help(SDS)

To be more specific and get help only for the get() method of the
SDS class:

  >>> help(SDS.get)   # or...
  >>> help(vinst.get) # if vinst is an SDS instance

pydoc can also be called from the command line, as in::

  % pydoc pyhdf.SD.SDS        # doc for the whole SDS class
  % pydoc pyhdf.SD.SDS.get    # doc for the SDS.get method

Summary of differences between the pyhdf and C SD API
-----------------------------------------------------
Most of the differences between the pyhdf and C SD API can
be summarized as follows.

- In the C API, every function returns an integer status code, and values
  computed by the function are returned through one or more pointers
  passed as arguments.
- In pyhdf, error statuses are returned through the Python exception
  mechanism, and values are returned as the method result. When the
  C API specifies that multiple values are returned, pyhdf returns a
  tuple of values, which are ordered similarly to the pointers in the
  C function argument list.

Error handling
--------------
All errors that the C SD API reports with a SUCCESS/FAIL error code
are reported by pyhdf using the Python exception mechanism.
When the C library reports a FAIL status, pyhdf raises an HDF4Error
exception (a subclass of Exception) with a descriptive message.
Unfortunately, the C library is rarely informative about the cause of
the error. pyhdf does its best to try to document the error, but most
of the time cannot do more than saying "execution error".

Attribute access: low and high level
------------------------------------
In the SD API, attributes can be of many types (integer, float, string,
etc) and can be single or multi-valued. Attributes can be set either at
the dataset, the variable or the dimension level. This can can be achieved
in two ways.

- By calling the get()/set() method of an attribute instance. In the
  following example, HDF file 'example.hdf' is created, and string
  attribute 'title' is attached to the file and given value
  'example'.

     >>> from pyhdf.SD import *
     >>> d = SD('example.hdf',SDC.WRITE|SDC.CREATE)  # create file
     >>> att = d.attr('title')            # create attribute instance
     >>> att.set(SDC.CHAR, 'example')     # set attribute type and value
     >>> print(att.get())                  # get attribute value
     >>>

- By handling the attribute like an ordinary Python class attribute.
  The above example can then be rewritten as follows:

     >>> from pyhdf.SD import *
     >>> d = SD('example.hdf',SDC.WRITE|SDC.CREATE)  # create dataset
     >>> d.title = 'example'              # set attribute type and value
     >>> print(d.title)                    # get attribute value
     >>>

What has been said above applies as well to multi-valued attributes.

    >>> att = d.attr('values')            # With an attribute instance
    >>> att.set(SDC.INT32, (1,2,3,4,5))   # Assign 5 ints as attribute value
    >>> att.get()                         # Get attribute values
    [1, 2, 3, 4, 5]

    >>> d.values = (1,2,3,4,5)            # As a Python class attribute
    >>> d.values                          # Get attribute values
    [1, 2, 3, 4, 5]

When the attribute is known by its name , standard functions 'setattr()'
and 'getattr()' can be used to replace the dot notation.
Above example becomes:

    >>> setattr(d, 'values', (1,2,3,4,5))
    >>> getattr(d, 'values')
    [1, 2, 3, 4, 5]

Handling a SD attribute like a Python class attribute is admittedly
more natural, and also much simpler. Some control is however lost in
doing so.

- Attribute type cannot be specified. pyhdf automatically selects one of
  three types according to the value(s) assigned to the attribute:
  SDC.CHAR if value is a string, SDC.INT32 if all values are integral,
  SDC.DOUBLE if one value is a float.
- Consequently, byte values cannot be assigned.
- Attribute properties (length, type, index number) can only be queried
  through methods of an attribute instance.

Variable access: low and high level
-----------------------------------
Similarly to attributes, datasets can be read/written in two ways.

The first way is through the get()/set() methods of a dataset instance.
Those methods accept parameters to specify the starting indices, the count
of values to read/write, and the strides along each dimension. For example,
if 'v' is a 4x4 array:

    >>> v.get()                         # complete array
    >>> v.get(start=(0,0),count=(1,4))  # first row
    >>> v.get(start=(0,1),count=(2,2),  # second and third columns of
    ...       stride=(2,1))             # first and third row

The second way is by indexing and slicing the variable like a Python
sequence. pyhdf here follows most of the rules used to index and slice
numpy arrays. Thus an HDF dataset can be seen almost as a numpy
array, except that data is read from/written to a file instead of memory.

Extended indexing let you access variable elements with the familiar
[i,j,...] notation, with one index per dimension. For example, if 'm' is a
rank 3 dataset, one could write:

    >>> m[0,3,5] = m[0,5,3]

When indexing is used to select a dimension in a 'get' operation, this
dimension is removed from the output array, thus reducing its rank by 1. A
rank 0 array is converted to a scalar. Thus, for a 3x3x3 'm' dataset
(rank 3) of integer type :

    >>> a = m[0]         # a is a 3x3 array (rank 2)
    >>> a = m[0,0]       # a is a 3 element array (rank 1)
    >>> a = m[0,0,0]     # a is an integer (rank 0 array becomes a scalar)

Had this rule not be followed, m[0,0,0] would have resulted in a single
element array, which could complicate computations.

Extended slice syntax allows slicing HDF datasets along each of its
dimensions, with the specification of optional strides to step through
dimensions at regular intervals. For each dimension, the slice syntax
is: "i:j[:stride]", the stride being optional. As with ordinary slices,
the starting and ending values of a slice can be omitted to refer to the
first and last element, respectively, and the end value can be negative to
indicate that the index is measured relative to the tail instead of the
beginning. Omitted dimensions are assumed to be sliced from beginning to
end. Thus:

    >>> m[0]             # treated as 'm[0,:,:]'.

Example above with get()/set() methods can thus be rewritten as follows:

    >>> v[:]             # complete array
    >>> v[:1]            # first row
    >>> v[::2,1:3]       # second and third columns of first and third row

Indexes and slices can be freely mixed, eg:

    >>> m[:2,3,1:3:2]

Note that, countrary to indexing, a slice never reduces the rank of the
output array, even if its length is 1. For example, given a 3x3x3 'm'
dataset:

    >>> a = m[0]         # indexing: a is a 3x3 array (rank 2)
    >>> a = m[0:1]       # slicing: a is a 1x3x3 array (rank 3)

As can easily be seen, extended slice syntax is much more elegant and
compact, and offers a few possibilities not easy to achieve with the
get()/sett() methods. Negative indices offer a nice example:

    >>> v[-2:]                         # last two rows
    >>> v[-3:-1]                       # second and third row
    >>> v[:,-1]                        # last column

Reading/setting multivalued HDF attributes and variables
--------------------------------------------------------
Multivalued HDF attributes are set using a python sequence (tuple or
list). Reading such an attribute returns a python list. The easiest way to
read/set an attribute is by handling it like a Python class attribute
(see "High level attribute access"). For example:

    >>> d=SD('test.hdf',SDC.WRITE|SDC.CREATE)  # create file
    >>> d.integers = (1,2,3,4)         # define multivalued integer attr
    >>> d.integers                     # get the attribute value
    [1, 2, 3, 4]

The easiest way to set multivalued HDF datasets is to assign to a
subset of the dataset, using "[:]" to assign to the whole dataset
(see "High level variable access"). The assigned value can be a python
sequence, which can be multi-leveled when assigning to a multdimensional
dataset. For example:

    >>> d=SD('test.hdf',SDC.WRITE|SDC.CREATE) # create file
    >>> v1=d.create('v1',SDC.INT32,3)         # 3-elem vector
    >>> v1[:]=[1,2,3]                         # assign 3-elem python list
    >>> v2=d.create('d2',SDC.INT32,(3,3))     # create 3x3 variable
           # The list assigned to v2 is composed
           # of 3 lists, each representing a row of v2.
    >>> v2[:]=[[1,2,3],[11,12,13],[21,22,23]]

The assigned value can also be a numpy array. Rewriting example above:

    >>> v1=array([1,2,3])
    >>> v2=array([[1,2,3],[11,12,13],[21,22,23]])

Note how we use indexing expressions 'v1[:]' and 'v2[:]' when assigning
using python sequences, and just the variable names when assigning numpy
arrays.

Reading an HDF dataset always returns a numpy array, except if
indexing is used and produces a rank-0 array, in which case a scalar is
returned.

netCDF files
------------
Files written in the popular Unidata netCDF format can be read and updated
using the HDF SD API. However, pyhdf cannot create netCDF formatted
files from scratch. The python 'pycdf' package can be used for that.

When accessing netCDF files through pyhdf, one should be aware of the
following differences between the netCDF and the HDF SD libraries.

- Differences in terminology can be confusing. What netCDF calls a
  'dataset' is called a 'file' or 'SD interface' in HDF. What HDF calls
  a dataset is called a 'variable' in netCDF parlance.
- In the netCDF API, dimensions are defined at the global (netCDF dataset)
  level. Thus, two netCDF variables defined over dimensions X and Y
  necessarily have the same rank and shape.
- In the HDF SD API, dimensions are defined at the HDF dataset level,
  except when they are named. Dimensions with the same name are considered
  to be "shared" between all the file datasets. They must be of the same
  length, and they share all their scales and attributes. For example,
  setting an attribute on a shared dimension affects all datasets sharing
  that dimension.
- When two or more netCDF variables are based on the unlimited dimension,
  they automatically grow in sync. If variables A and B use the unlimited
  dimension, adding "records" to A along its unlimited dimension
  implicitly adds records in B (which are left in an undefined state and
  filled with the fill_value when the file is refreshed).
- In HDF, unlimited dimensions behave independently. If HDF datasets A and
  B are based on an unlimited dimension, adding records to A does not
  affect the number of records to B. This is true even if the unlimited
  dimensions bear the same name (they do not appear to be "shared" as is
  the case when the dimensions are fixed).


Classes summary
---------------
pyhdf wraps the SD API using different types of python classes::

  SD     HDF SD interface (almost synonymous with the subset of the
         HDF file holding all the SD datasets)
  SDS    scientific dataset
  SDim   dataset dimension
  SDAttr attribute (either at the file, dataset or dimension level)
  SDC    constants (opening modes, data types, etc)

In more detail::

  SD     The SD class implements the HDF SD interface as applied to a given
         file. This class encapsulates the "SD interface" identifier
         (referred to as "sd_id" in the C API documentation), and all
         the SD API top-level functions.

         To create an SD instance, call the SD() constructor.

         methods:
           constructors:
             SD()          open an existing HDF file or create a new one,
                           returning an SD instance
             attr()        create an SDAttr (attribute) instance to access
                           an existing file attribute or create a new one;
                           "dot notation" can also be used to get and set
                           an attribute
             create()      create a new dataset, returning an SDS instance
             select()      locate an existing dataset given its name or
                           index number, returning an SDS instance

           file closing
             end()         end access to the SD interface and close the
                           HDF file

           inquiry
             attributes()  return a dictionnary describing every global
                           attribute attached to the HDF file
             datasets()    return a dictionnary describing every dataset
                           stored inside the file
             info()        get the number of datasets stored in the file
                           and the number of attributes attached to it
             nametoindex() get a dataset index number given the dataset
                           name
             reftoindex()  get a dataset index number given the dataset
                           reference number

           misc
             setfillmode() set the fill mode for all the datasets in
                           the file


  SDAttr The SDAttr class defines an attribute, either at the file (SD),
         dataset (SDS) or dimension (SDim) level. The class encapsulates
         the object to which the attribute is attached, and the attribute
         name.

         To create an SDAttr instance, obtain an instance for an SD (file),
         SDS (dataset) or dimension (SDim) object, and call its attr()
         method.

         NOTE. An attribute can also be read/written like
               a python class attribute, using the familiar
               dot notation. See "High level attribute access".

         methods:
           read/write value
             get()         get the attribute value
             set()         set the attribute value


           inquiry
             index()       get the attribute index number
             info()        get the attribute name, type and number of
                           values


  SDC    The SDC class holds contants defining file opening modes and
         data types. Constants are named after their C API counterparts.

           file opening modes:
             SDC.CREATE      create file if non existent
             SDC.READ        read-only mode
             SDC.TRUNC       truncate file if already exists
             SDC.WRITE       read-write mode

           data types:
             SDC.CHAR        8-bit character
             SDC.CHAR8       8-bit character
             SDC.UCHAR       unsigned 8-bit integer
             SDC.UCHAR8      unsigned 8-bit integer
             SDC.INT8        signed 8-bit integer
             SDC.UINT8       unsigned 8-bit integer
             SDC.INT16       signed 16-bit integer
             SDC.UINT16      unsigned 16-bit intege
             SDC.INT32       signed 32-bit integer
             SDC.UINT32      unsigned 32-bit integer
             SDC.FLOAT32     32-bit floating point
             SDC.FLOAT64     64-bit floaring point

           dataset fill mode:
             SDC.FILL
             SDC.NOFILL

           dimension:
             SDC.UNLIMITED   dimension can grow dynamically

           data compression:
             SDC.COMP_NONE
             SDC.COMP_RLE
             SDC.COMP_NBIT
             SDC.COMP_SKPHUFF
             SDC.COMP_DEFLATE
             SDC.COMP_SZIP
             SDC.COMP_SZIP_EC
             SDC.COMP_SZIP_NN
             SDC.COMP_SZIP_RAW

  SDS    The SDS class implements an HDF scientific dataset (SDS) object.

         To create an SDS instance, call the create() or select() methods
         of an SD instance.

         methods:
           constructors
             attr()        create an SDAttr (attribute) instance to access
                           an existing dataset attribute or create a
                           new one; "dot notation" can also be used to get
                           and set an attribute

             dim()         return an SDim (dimension) instance for a given
                           dataset dimension, given the dimension index
                           number

           dataset closing
             endaccess()   terminate access to the dataset

           inquiry
             attributes()  return a dictionnary describing every
                           attribute defined on the dataset
             checkempty()  determine whether the dataset is empty
             dimensions()  return a dictionnary describing all the
                           dataset dimensions
             info()        get the dataset name, rank, dimension lengths,
                           data type and number of attributes
             iscoordvar()  determine whether the dataset is a coordinate
                           variable (holds a dimension scale)
             isrecord()    determine whether the dataset is appendable
                           (the dataset dimension 0 is unlimited)
             ref()         get the dataset reference number


           reading/writing data values
             get()         read data from the dataset
             set()         write data to the dataset

                           A dataset can also be read/written using the
                           familiar index and slice notation used to
                           access python sequences. See "High level
                           variable access".

           reading/writing  standard attributes
             getcal()       get the dataset calibration coefficients:
                              scale_factor, scale_factor_err, add_offset,
                              add_offset_err, calibrated_nt
             getdatastrs()  get the dataset standard string attributes:
                              long_name, units, format, coordsys
             getfillvalue() get the dataset fill value:
                              _FillValue
             getrange()     get the dataset min and max values:
                              valid_range
             setcal()       set the dataset calibration coefficients
             setdatastrs()  set the dataset standard string attributes
             setfillvalue() set the dataset fill value
             setrange()     set the dataset min and max values

           compression
             getcompress()  get info about the dataset compression type and mode
             setcompress()  set the dataset compression type and mode

           misc
             setexternalfile()  store the dataset in an external file

  SDim   The SDdim class implements a dimension object.

         To create an SDim instance, call the dim() method of an SDS
         (dataset) instance.

         Methods:
           constructors
             attr()         create an SDAttr (attribute) instance to access
                            an existing dimension attribute or create a
                            new one; "dot notation" can also be used to
                            get and set an attribute

           inquiry
             attributes()   return a dictionnary describing every
                            attribute defined on the dimension
             info()         get the dimension name, length, scale data type
                            and number of attributes
             length()       return the current dimension length

           reading/writing dimension data
             getscale()     get the dimension scale values
             setname()      set the dimension name
             setscale()     set the dimension scale values

           reading/writing standard attributes
             getstrs()      get the dimension standard string attributes:
                              long_name, units, format
             setstrs()      set the dimension standard string attributes

Data types
----------
Data types come into play when first defining datasets and their attributes,
and later when querying the definition of those objects.
Data types are specified using the symbolic constants defined inside the
SDC class of the SD module.

- CHAR and CHAR8 (equivalent): an 8-bit character.
- UCHAR, UCHAR8 and UINT8 (equivalent): unsigned 8-bit values (0 to 255)
- INT8:    signed 8-bit values (-128 to 127)
- INT16:   signed 16-bit values
- UINT16:  unsigned 16 bit values
- INT32:   signed 32 bit values
- UINT32:  unsigned 32 bit values
- FLOAT32: 32 bit floating point values (C floats)
- FLOAT64: 64 bit floating point values (C doubles)

There is no explicit "string" type. To simulate a string, set the
type to CHAR, and set the length to a value of 'n' > 1. This creates and
"array of characters", close to a string (except that strings will always
be of length 'n', right-padded with spaces if necessary).


Programming models
------------------

Writing
^^^^^^^
The following code can be used as a model to create an SD dataset.
It shows how to use the most important functionnalities
of the SD interface needed to initialize a dataset.
A real program should of course add error handling::

    # Import SD and numpy.
    from pyhdf.SD import *
    from numpy import *

    fileName = 'template.hdf'
    # Create HDF file.
    hdfFile = SD(fileName ,SDC.WRITE|SDC.CREATE)
    # Assign a few attributes at the file level
    hdfFile.author = 'It is me...'
    hdfFile.priority = 2
    # Create a dataset named 'd1' to hold a 3x3 float array.
    d1 = hdfFile.create('d1', SDC.FLOAT32, (3,3))
    # Set some attributs on 'd1'
    d1.description = 'Sample 3x3 float array'
    d1.units = 'celsius'
    # Name 'd1' dimensions and assign them attributes.
    dim1 = d1.dim(0)
    dim2 = d1.dim(1)
    dim1.setname('width')
    dim2.setname('height')
    dim1.units = 'm'
    dim2.units = 'cm'
    # Assign values to 'd1'
    d1[0]  = (14.5, 12.8, 13.0)  # row 1
    d1[1:] = ((-1.3, 0.5, 4.8),  # row 2 and
              (3.1, 0.0, 13.8))  # row 3
    # Close dataset
    d1.endaccess()
    # Close file
    hdfFile.end()

Reading
^^^^^^^
The following code, which reads the dataset created above, can also serve as
a model for any program which needs to access an SD dataset::

    # Import SD and numpy.
    from pyhdf.SD import *
    from numpy import *

    fileName = 'template.hdf'
    # Open file in read-only mode (default)
    hdfFile = SD(fileName)
    # Display attributes.
    print "file:", fileName
    print "author:", hdfFile.author
    print "priority:", hdfFile.priority
    # Open dataset 'd1'
    d1 = hdfFile.select('d1')
    # Display dataset attributes.
    print "dataset:", 'd1'
    print "description:",d1.description
    print "units:", d1.units
    # Display dimensions info.
    dim1 = d1.dim(0)
    dim2 = d1.dim(1)
    print "dimensions:"
    print "dim1: name=", dim1.info()[0],
    print "length=", dim1.length(),
    print "units=", dim1.units
    print "dim2: name=", dim2.info()[0],
    print "length=", dim2.length(),
    print "units=", dim2.units
    # Show dataset values
    print d1[:]
    # Close dataset
    d1.endaccess()
    # Close file
    hdfFile.end()


Examples
--------

Example-1
^^^^^^^^^
The following simple example exercises some important pyhdf.SD methods. It
shows how to create an HDF dataset, define attributes and dimensions,
create variables, and assign their contents.

Suppose we have a series of text files each defining a 2-dimensional real-
valued matrix. First line holds the matrix dimensions, and following lines
hold matrix values, one row per line. The following procedure will load
into an HDF dataset the contents of any one of those text files. The
procedure computes the matrix min and max values, storing them as
dataset attributes. It also assigns to the variable the group of
attributes passed as a dictionnary by the calling program. Note how simple
such an assignment becomes with pyhdf: the dictionnary can contain any
number of attributes, of different types, single or multi-valued. Doing
the same in a conventional language would be a much more challenging task.

Error checking is minimal, to keep example as simple as possible
(admittedly a rather poor excuse ...)::

    from numpy import *
    from pyhdf.SD import *

    import os

    def txtToHDF(txtFile, hdfFile, varName, attr):

        try:  # Catch pyhdf errors
            # Open HDF file in update mode, creating it if non existent.
            d = SD(hdfFile, SDC.WRITE|SDC.CREATE)
            # Open text file and get matrix dimensions on first line.
            txt = open(txtFile)
            ni, nj = map(int, txt.readline().split())
            # Define an HDF dataset of 32-bit floating type (SDC.FLOAT32)
            # with those dimensions.
            v = d.create(varName, SDC.FLOAT32, (ni, nj))
            # Assign attributes passed as argument inside dict 'attr'.
            for attrName in attr.keys():
                setattr(v, attrName, attr[attrName])
            # Load variable with lines of data. Compute min and max
            # over the whole matrix.
            i = 0
            while i < ni:
                elems = map(float, txt.readline().split())
                v[i] = elems  # load row i
                minE = min(elems)
                maxE = max(elems)
                if i:
                    minVal = min(minVal, minE)
                    maxVal = max(maxVal, maxE)
                else:
                    minVal = minE
                    maxVal = maxE
                i += 1
            # Set variable min and max attributes.
            v.minVal = minVal
            v.maxVal = maxVal
            # Close dataset and file objects (not really necessary, since
            # closing is automatic when objects go out of scope.
            v.endaccess()
            d.end()
            txt.close()
        except HDF4Error, msg:
            print "HDF4Error:", msg


We could now call the procedure as follows::

    hdfFile  = 'table.hdf'
    try:  # Delete if exists.
        os.remove(hdfFile)
    except:
        pass
    # Load contents of file 'temp.txt' into dataset 'temperature'
    # an assign the attributes 'title', 'units' and 'valid_range'.
    txtToHDF('temp.txt', hdfFile, 'temperature',
             {'title'      : 'temperature matrix',
              'units'      : 'celsius',
              'valid_range': (-2.8,27.0)})

    # Load contents of file 'depth.txt' into dataset 'depth'
    # and assign the same attributes as above.
    txtToHDF('depth.txt', hdfFile, 'depth',
             {'title'      : 'depth matrix',
              'units'      : 'meters',
              'valid_range': (0, 500.0)})


Example 2
^^^^^^^^^
This example shows a usefull python program that will display the
structure of the SD component of any HDF file whose name is given on
the command line. After the HDF file is opened, high level inquiry methods
are called to obtain dictionnaries descrybing attributes, dimensions and
datasets. The rest of the program mostly consists in nicely formatting
the contents of those dictionaries::

    import sys
    from pyhdf.SD import *
    from numpy import *

    # Dictionnary used to convert from a numeric data type to its symbolic
    # representation
    typeTab = {
               SDC.CHAR:    'CHAR',
               SDC.CHAR8:   'CHAR8',
               SDC.UCHAR8:  'UCHAR8',
               SDC.INT8:    'INT8',
               SDC.UINT8:   'UINT8',
               SDC.INT16:   'INT16',
               SDC.UINT16:  'UINT16',
               SDC.INT32:   'INT32',
               SDC.UINT32:  'UINT32',
               SDC.FLOAT32: 'FLOAT32',
               SDC.FLOAT64: 'FLOAT64'
               }

    printf = sys.stdout.write

    def eol(n=1):
        printf("%s" % chr(10) * n)

    hdfFile = sys.argv[1]    # Get first command line argument

    try:  # Catch pyhdf.SD errors
      # Open HDF file named on the command line
      f = SD(hdfFile)
      # Get global attribute dictionnary
      attr = f.attributes(full=1)
      # Get dataset dictionnary
      dsets = f.datasets()

      # File name, number of attributes and number of variables.
      printf("FILE INFO"); eol()
      printf("-------------"); eol()
      printf("%-25s%s" % ("File:", hdfFile)); eol()
      printf("%-25s%d" % ("  file attributes:", len(attr))); eol()
      printf("%-25s%d" % ("  datasets:", len(dsets))); eol()
      eol();

      # Global attribute table.
      if len(attr) > 0:
          printf("File attributes"); eol(2)
          printf("  name                 idx type    len value"); eol()
          printf("  -------------------- --- ------- --- -----"); eol()
          # Get list of attribute names and sort them lexically
          attNames = attr.keys()
          attNames.sort()
          for name in attNames:
              t = attr[name]
                  # t[0] is the attribute value
                  # t[1] is the attribute index number
                  # t[2] is the attribute type
                  # t[3] is the attribute length
              printf("  %-20s %3d %-7s %3d %s" %
                     (name, t[1], typeTab[t[2]], t[3], t[0])); eol()
          eol()


      # Dataset table
      if len(dsets) > 0:
          printf("Datasets (idx:index num, na:n attributes, cv:coord var)"); eol(2)
          printf("  name                 idx type    na cv dimension(s)"); eol()
          printf("  -------------------- --- ------- -- -- ------------"); eol()
          # Get list of dataset names and sort them lexically
          dsNames = dsets.keys()
          dsNames.sort()
          for name in dsNames:
              # Get dataset instance
              ds = f.select(name)
              # Retrieve the dictionary of dataset attributes so as
              # to display their number
              vAttr = ds.attributes()
              t = dsets[name]
                  # t[0] is a tuple of dimension names
                  # t[1] is a tuple of dimension lengths
                  # t[2] is the dataset type
                  # t[3] is the dataset index number
              printf("  %-20s %3d %-7s %2d %-2s " %
                     (name, t[3], typeTab[t[2]], len(vAttr),
                      ds.iscoordvar() and 'X' or ''))
              # Display dimension info.
              n = 0
              for d in t[0]:
                  printf("%s%s(%d)" % (n > 0 and ', ' or '', d, t[1][n]))
                  n += 1
              eol()
          eol()

      # Dataset info.
      if len(dsNames) > 0:
          printf("DATASET INFO"); eol()
          printf("-------------"); eol(2)
          for name in dsNames:
              # Access the dataset
              dsObj = f.select(name)
              # Get dataset attribute dictionnary
              dsAttr = dsObj.attributes(full=1)
              if len(dsAttr) > 0:
                  printf("%s attributes" % name); eol(2)
                  printf("  name                 idx type    len value"); eol()
                  printf("  -------------------- --- ------- --- -----"); eol()
                  # Get the list of attribute names and sort them alphabetically.
                  attNames = dsAttr.keys()
                  attNames.sort()
                  for nm in attNames:
                      t = dsAttr[nm]
                          # t[0] is the attribute value
                          # t[1] is the attribute index number
                          # t[2] is the attribute type
                          # t[3] is the attribute length
                      printf("  %-20s %3d %-7s %3d %s" %
                             (nm, t[1], typeTab[t[2]], t[3], t[0])); eol()
                  eol()
              # Get dataset dimension dictionnary
              dsDim = dsObj.dimensions(full=1)
              if len(dsDim) > 0:
                  printf ("%s dimensions" % name); eol(2)
                  printf("  name                 idx len   unl type    natt");eol()
                  printf("  -------------------- --- ----- --- ------- ----");eol()
                  # Get the list of dimension names and sort them alphabetically.
                  dimNames = dsDim.keys()
                  dimNames.sort()
                  for nm in dimNames:
                      t = dsDim[nm]
                          # t[0] is the dimension length
                          # t[1] is the dimension index number
                          # t[2] is 1 if the dimension is unlimited, 0 if not
                          # t[3] is the the dimension scale type, 0 if no scale
                          # t[4] is the number of attributes
                      printf("  %-20s %3d %5d  %s  %-7s %4d" %
                             (nm, t[1], t[0], t[2] and "X" or " ",
                              t[3] and typeTab[t[3]] or "", t[4])); eol()
                  eol()


    except HDF4Error, msg:
        print "HDF4Error", msg



"""
import os, sys, types

from . import hdfext as _C
from .six.moves import xrange
from .error import _checkErr, HDF4Error

# List of names we want to be imported by an "from pyhdf.SD import *"
# statement

__all__ = ['SD', 'SDAttr', 'SDC', 'SDS', 'SDim', 'HDF4Error']

try:
    import numpy as _toto
    del _toto
except ImportError:
    raise HDF4Error("numpy package required but not installed")

class SDC(object):
    """The SDC class holds contants defining opening modes and data types.

           file opening modes:
             ==========   ===    ===============================
             SDC.CREATE     4    create file if non existent
             SDC.READ       1    read-only mode
             SDC.TRUNC    256    truncate file if already exists
             SDC.WRITE      2    read-write mode
             ==========   ===    ===============================

           data types:
             ===========  ===    ===============================
             SDC.CHAR       4    8-bit character
             SDC.CHAR8      4    8-bit character
             SDC.UCHAR      3    unsigned 8-bit integer
             SDC.UCHAR8     3    unsigned 8-bit integer
             SDC.INT8      20    signed 8-bit integer
             SDC.UINT8     21    unsigned 8-bit integer
             SDC.INT16     22    signed 16-bit integer
             SDC.UINT16    23    unsigned 16-bit intege
             SDC.INT32     24    signed 32-bit integer
             SDC.UINT32    25    unsigned 32-bit integer
             SDC.FLOAT32    5    32-bit floating point
             SDC.FLOAT64    6    64-bit floaring point
             ===========  ===    ===============================

           dataset fill mode:
             ===========  ===
             SDC.FILL       0
             SDC.NOFILL   256
             ===========  ===

           dimension:
             =============  ===  ===============================
             SDC.UNLIMITED  0    dimension can grow dynamically
             =============  ===  ===============================

           data compression:
             =================  ===
             SDC.COMP_NONE      0
             SDC.COMP_RLE       1
             SDC.COMP_NBIT      2
             SDC.COMP_SKPHUFF   3
             SDC.COMP_DEFLATE   4
             SDC.COMP_SZIP      5

             SDC.COMP_SZIP_EC     4
             SDC.COMP_SZIP_NN    32
             SDC.COMP_SZIP_RAW  128
             =================  ===

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

    FILL         = _C.SD_FILL
    NOFILL       = _C.SD_NOFILL

    UNLIMITED    = _C.SD_UNLIMITED

    COMP_NONE    = _C.COMP_CODE_NONE
    COMP_RLE     = _C.COMP_CODE_RLE
    COMP_NBIT    = _C.COMP_CODE_NBIT
    COMP_SKPHUFF = _C.COMP_CODE_SKPHUFF
    COMP_DEFLATE = _C.COMP_CODE_DEFLATE
    COMP_SZIP    = _C.COMP_CODE_SZIP

    COMP_SZIP_EC  =   4
    COMP_SZIP_NN  =  32
    COMP_SZIP_RAW = 128

    # Types with an equivalent in the numpy package
    # NOTE:
    #  CHAR8 and INT8 are handled similarly (signed byte -128,...,0,...127)
    #  UCHAR8 and UINT8 are treated equivalently (unsigned byte: 0,1,...,255)
    #  UINT16 and UINT32 are supported
    #  INT64 and UINT64 are not yet supported py pyhdf
    equivNumericTypes = [FLOAT32, FLOAT64,
                         INT8, UINT8,
                         INT16, UINT16,
                         INT32, UINT32,
                         CHAR8, UCHAR8]

class SDAttr(object):

    def __init__(self, obj, index_or_name):
        """Init an SDAttr instance. Should not be called directly by
        the user program. An SDAttr instance must be created through
        the attr() methods of the SD, SDS or SDim classes.
                                                """
        # Args
        #  obj   object instance to which the attribute refers
        #        (SD, SDS, SDDim)
        #  index_or_name attribute index or name
        #
        # Class private attributes:
        #  _obj   object instance
        #  _index attribute index or None
        #  _name  attribute name or None

        self._obj = obj
        # Name is given, may exist or not.
        if isinstance(index_or_name, type('')):
            self._name = index_or_name
            self._index = None
        # Index is given. Must exist.
        else:
            self._index = index_or_name
            status, self._name, data_type, n_values = \
                    _C.SDattrinfo(self._obj._id, self._index)
            _checkErr('set', status, 'illegal attribute index')

    def info(self):
        """Retrieve info about the attribute : name, data type and
        number of values.

        Args::

          no argument

        Returns::

          3-element tuple holding:

          - attribute name
          - attribute data type (see constants SDC.xxx)
          - number of values in the attribute; for a string-valued
            attribute (data type SDC.CHAR8), the number of values
            corresponds to the string length


        C library equivalent : SDattrinfo
                                                       """
        if self._index is None:
            try:
                self._index = self._obj.findattr(self._name)
            except HDF4Error:
                raise HDF4Error("info: cannot convert name to index")
        status, self._name, data_type, n_values = \
                              _C.SDattrinfo(self._obj._id, self._index)
        _checkErr('info', status, 'illegal attribute index')
        return self._name, data_type, n_values

    def index(self):
        """Retrieve the attribute index number.

        Args::

          no argument

        Returns::

          attribute index number (starting at 0)

        C library equivalent : SDfindattr
                                             """

        self._index = _C.SDfindattr(self._obj._id, self._name)
        _checkErr('find', self._index, 'illegal attribute name')
        return self._index

    def get(self):
        """Retrieve the attribute value.

        Args::

          no argument

        Returns::

          attribute value(s); a list is returned if the attribute
          is made up of more than one value, except in the case of a
          string-valued attribute (data type SDC.CHAR8) where the
          values are returned as a string

        C library equivalent : SDreadattr

        Attributes can also be read like ordinary python attributes,
        using the dot notation. See "High level attribute access".

                                                """

        if self._index is None:
            try:
                self._index = self._obj.findattr(self._name)
            except HDF4Error:
                raise HDF4Error("get: cannot convert name to index")

        # Obtain attribute type and the number of values.
        status, self._name, data_type, n_values = \
                    _C.SDattrinfo(self._obj._id, self._index)
        _checkErr('read', status, 'illegal attribute index')

        # Get attribute value.
        convert = _array_to_ret
        if data_type == SDC.CHAR8:
            buf = _C.array_byte(n_values)
            convert = _array_to_str

        elif data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            buf = _C.array_int8(n_values)

        elif data_type == SDC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("read: attribute index %d has an "\
                             "illegal or unupported type %d" % \
                             (self._index, data_type))

        status = _C.SDreadattr(self._obj._id, self._index, buf)
        _checkErr('read', status, 'illegal attribute index')
        return convert(buf, n_values)

    def set(self, data_type, values):
        """Update/Create a new attribute and set its value(s).

        Args::

          data_type    : attribute data type (see constants SDC.xxx)
          values       : attribute value(s); specify a list to create
                         a multi-valued attribute; a string valued
                         attribute can be created by setting 'data_type'
                         to SDC.CHAR8 and 'values' to the corresponding
                         string

        Returns::

          None

        C library equivalent : SDsetattr

        Attributes can also be written like ordinary python attributes,
        using the dot notation. See "High level attribute access".

                                                  """
        try:
            n_values = len(values)
        except:
            n_values = 1
            values = [values]
        if data_type == SDC.CHAR8:
            buf = _C.array_byte(n_values)
            # Allow values to be passed as a string.
            # Noop if a list is passed.
            values = list(values)
            for n in range(n_values):
                values[n] = ord(values[n])

        elif data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            # SWIG refuses negative values here. We found that if we
            # pass them as byte values, it will work.
            buf = _C.array_int8(n_values)
            values = list(values)
            for n in range(n_values):
                v = values[n]
                if v >= 0:
                    v &= 0x7f
                else:
                    v = abs(v) & 0x7f
                    if v:
                        v = 256 - v
                    else:
                        v = 128         # -128 in 2s complement
                values[n] = v

        elif data_type == SDC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("set: illegal or unimplemented data_type")

        for n in range(n_values):
            buf[n] = values[n]
        status = _C.SDsetattr(self._obj._id, self._name,
                              data_type, n_values, buf)
        _checkErr('set', status, 'illegal attribute')
        # Init index following attribute creation.
        self._index = _C.SDfindattr(self._obj._id, self._name)
        _checkErr('find', self._index, 'illegal attribute')


class SD(object):
    """The SD class implements an HDF SD interface.
    To instantiate an SD class, call the SD() constructor.
    To set attributes on an SD instance, call the SD.attr()
    method to create an attribute instance, then call the methods
    of this instance. """


    def __init__(self, path, mode=SDC.READ):
        """SD constructor. Initialize an SD interface on an HDF file,
        creating the file if necessary.

        Args::

          path    name of the HDF file on which to open the SD interface
          mode    file opening mode; this mode is a set of binary flags
                  which can be ored together

                      SDC.CREATE  combined with SDC.WRITE to create file
                                  if it does not exist
                      SDC.READ    open file in read-only access (default)
                      SDC.TRUNC   if combined with SDC.WRITE, overwrite
                                  file if it already exists
                      SDC.WRITE   open file in read-write mode; if file
                                  exists it is updated, unless SDC.TRUNC is
                                  set, in which case it is erased and
                                  recreated; if file does not exist, an
                                  error is raised unless SDC.CREATE is set,
                                  in which case the file is created

                   Note an important difference in the way CREATE is
                   handled by the C library and the pyhdf package.
                   For the C library, CREATE indicates that a new file
                   should always be created, overwriting an existing one if
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

        Returns::

          an SD instance

        C library equivalent : SDstart
                                                     """
        # Private attributes:
        #  _id:       file id

        # Make sure _id is initialized in case __del__ is called
        # when the SD object goes out of scope after failing to
        # open file. Failure to do so may put python into an infinite loop
        # (thanks to Richard.Andrews@esands.com for reporting this bug).
        self._id = None

        # See if file exists.
        exists = os.path.exists(path)

        # We must have either WRITE or READ flag.
        if SDC.WRITE & mode:
            if exists:
                if SDC.TRUNC & mode:
                    try:
                        os.remove(path)
                    except Exception as msg:
                        raise HDF4Error(msg)
                    mode = SDC.CREATE|SDC.WRITE
                else:
                    mode = SDC.WRITE
            else:
                if SDC.CREATE & mode:
                    mode |= SDC.WRITE
                else:
                    raise HDF4Error("SD: no such file")
        elif SDC.READ & mode:
            if exists:
                mode = SDC.READ
            else:
                raise HDF4Error("SD: no such file")
        else:
            raise HDF4Error("SD: bad mode, READ or WRITE must be set")

        id = _C.SDstart(path, mode)
        _checkErr('SD', id, "cannot open %s" % path)
        self._id = id


    def __del__(self):
        """Delete the instance, first calling the end() method
        if not already done.          """

        try:
            if self._id:
                self.end()
        except:
            pass

    def __getattr__(self, name):
        # Get value(s) of SD attribute 'name'.

        return _getattr(self, name)

    def __setattr__(self, name, value):
        # Set value(s) of SD attribute 'name'.

        # A name starting with an underscore will be treated as
        # a standard python attribute, and as an HDF attribute
        # otherwise.

        _setattr(self, name, value, ['_id'])

    def end(self):
        """End access to the SD interface and close the HDF file.

        Args::

            no argument

        Returns::

            None

        The instance should not be used afterwards.
        The 'end()' method is implicitly called when the
        SD instance is deleted.

        C library equivalent : SDend
                                                      """

        status = _C.SDend(self._id)
        _checkErr('end', status, "cannot execute")
        self._id = None

    def info(self):
        """Retrieve information about the SD interface.

        Args::

          no argument

        Returns::

          2-element tuple holding:
            number of datasets inside the file
            number of file attributes

        C library equivalent : SDfileinfo
                                                  """

        status, n_datasets, n_file_attrs = _C.SDfileinfo(self._id)
        _checkErr('info', status, "cannot execute")
        return n_datasets, n_file_attrs

    def nametoindex(self, sds_name):
        """Return the index number of a dataset given the dataset name.

        Args::

          sds_name  : dataset name

        Returns::

          index number of the dataset

        C library equivalent : SDnametoindex
                                                 """

        sds_idx = _C.SDnametoindex(self._id, sds_name)
        _checkErr('nametoindex', sds_idx, 'non existent SDS')
        return sds_idx

    def reftoindex(self, sds_ref):
        """Returns the index number of a dataset given the dataset
        reference number.

        Args::

          sds_ref : dataset reference number

        Returns::

          dataset index number

        C library equivalent : SDreftoindex
                                             """

        sds_idx = _C.SDreftoindex(self._id, sds_ref)
        _checkErr('reftoindex', sds_idx, 'illegal SDS ref number')
        return sds_idx

    def setfillmode(self, fill_mode):
        """Set the fill mode for all the datasets in the file.

        Args::

          fill_mode : fill mode; one of :
                        SDC.FILL   write the fill value to all the datasets
                                  of the file by default
                        SDC.NOFILL do not write fill values to all datasets
                                  of the file by default

        Returns::

          previous fill mode value

        C library equivalent: SDsetfillmode
                                                            """

        if not fill_mode in [SDC.FILL, SDC.NOFILL]:
            raise HDF4Error("bad fill mode")
        old_mode = _C.SDsetfillmode(self._id, fill_mode)
        _checkErr('setfillmode', old_mode, 'cannot execute')
        return old_mode

    def create(self, name, data_type, dim_sizes):
        """Create a dataset.

        Args::

          name           dataset name
          data_type      type of the data, set to one of the SDC.xxx
                         constants;
          dim_sizes      lengths of the dataset dimensions; a one-
                         dimensional array is specified with an integer,
                         an n-dimensional array with an n-element sequence
                         of integers; the length of the first dimension can
                         be set to SDC.UNLIMITED to create an unlimited
                         dimension (a "record" variable).

                         IMPORTANT:  netCDF and HDF differ in the way
                         the UNLIMITED dimension is handled. In netCDF,
                         all variables of a dataset with an unlimited
                         dimension grow in sync, eg adding a record to
                         a variable will implicitly extend other record
                         variables. In HDF, each record variable grows
                         independently of each other.

        Returns::

          SDS instance for the dataset

        C library equivalent : SDcreate

                                                                    """

        # Validate args.
        if isinstance(dim_sizes, type(1)):  # allow k instead of [k]
                                        # for a 1-dim arr
            dim_sizes = [dim_sizes]
        rank = len(dim_sizes)
        buf = _C.array_int32(rank)
        for n in range(rank):
            buf[n] = dim_sizes[n]
        id = _C.SDcreate(self._id, name, data_type, rank, buf)
        _checkErr('CREATE', id, "cannot execute")
        return SDS(self, id)

    def select(self, name_or_index):
        """Locate a dataset.

        Args::

          name_or_index  dataset name or index number

        Returns::

          SDS instance for the dataset

        C library equivalent : SDselect
                                                                    """

        if isinstance(name_or_index, type(1)):
            idx = name_or_index
        else:
            try:
                idx = self.nametoindex(name_or_index)
            except HDF4Error:
                raise HDF4Error("select: non-existent dataset")
        id = _C.SDselect(self._id, idx)
        _checkErr('select', id, "cannot execute")
        return SDS(self, id)

    def attr(self, name_or_index):
        """Create an SDAttr instance representing a global
        attribute (defined at the level of the SD interface).

        Args::

          name_or_index   attribute name or index number; if a name is
                          given, the attribute may not exist; in that
                          case, it will be created when the SDAttr
                          instance set() method is called

        Returns::

          SDAttr instance for the attribute. Call the methods of this
          class to query, read or set the attribute.

        C library equivalent : no equivalent

                                """

        return SDAttr(self, name_or_index)


    def attributes(self, full=0):
        """Return a dictionnary describing every global
        attribute attached to the SD interface.

        Args::

          full      true to get complete info about each attribute
                    false to report only each attribute value

        Returns::

          Empty dictionnary if no global attribute defined
          Otherwise, dictionnary where each key is the name of a
          global attribute. If parameter 'full' is false,
          key value is the attribute value. If 'full' is true,
          key value is a tuple with the following elements:

          - attribute value
          - attribute index number
          - attribute type
          - attribute length

        C library equivalent : no equivalent
                                                    """

        # Get the number of global attributes.
        nsds, natts = self.info()

        # Inquire each attribute
        res = {}
        for n in range(natts):
            a = self.attr(n)
            name, aType, nVal = a.info()
            if full:
                res[name] = (a.get(), a.index(), aType, nVal)
            else:
                res[name] = a.get()

        return res

    def datasets(self):
        """Return a dictionnary describing all the file datasets.

        Args::

          no argument

        Returns::

          Empty dictionnary if no dataset is defined.
          Otherwise, dictionnary whose keys are the file dataset names,
          and values are tuples describing the corresponding datasets.
          Each tuple holds the following elements in order:

          - tuple holding the names of the dimensions defining the
            dataset coordinate axes
          - tuple holding the dataset shape (dimension lengths);
            if a dimension is unlimited, the reported length corresponds
            to the dimension current length
          - dataset type
          - dataset index number

        C library equivalent : no equivalent
                                                """
        # Get number of datasets
        nDs = self.info()[0]

        # Inquire each var
        res = {}
        for n in range(nDs):
            # Get dataset info.
            v = self.select(n)
            vName, vRank, vLen, vType, vAtt = v.info()
            if vRank < 2:     # need a sequence
                vLen = [vLen]
            # Get dimension info.
            dimNames = []
            dimLengths = []
            for dimNum in range(vRank):
                d = v.dim(dimNum)
                dimNames.append(d.info()[0])
                dimLengths.append(vLen[dimNum])
            res[vName] = (tuple(dimNames), tuple(dimLengths),
                         vType, n)

        return res


class SDS(object):
    """The SDS class implements an HDF dataset object.
    To create an SDS instance, call the create() or select()
    methods of the SD class. To set attributes on an SDS instance,
    call the SDS.attr() method to create an attribute instance,
    then call the methods of this instance. Attributes can also be
    set using the "dot notation". """

    def __init__(self, sd, id):
        """This constructor should not be called by the user program.
        Call the SD.create() and SD.select() methods instead.
                                                  """

        # Args
        #  sd   : SD instance
        #  id   : SDS identifier


        # Private attributes
        #  _sd  SD intance
        #  _id  SDS identifier
        self._sd = sd
        self._id = id

    def __del__(self):

        # Delete the instance, first calling the endaccess() method
        # if not already done.

        try:
            if self._id:
                self.endaccess()
        except:
            pass

    def __getattr__(self, name):
        # Get value(s) of SDS attribute 'name'.

        return _getattr(self, name)

    def __setattr__(self, name, value):
        # Set value(s) of SDS attribute 'name'.

        _setattr(self, name, value, ['_sd', '_id'])

    def __len__(self):    # Needed for slices like "-2:" but why ?

        return 0

    def __getitem__(self, elem):

        # This special method is used to index the SDS dataset
        # using the "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the SDreaddara() function, and is much more easy
        # to use.

        # Compute arguments to 'SDreaddata_0()'.
        start, count, stride = self.__buildStartCountStride(elem)
        # Get elements.
        return self.get(start, count, stride)

    def __setitem__(self, elem, data):

        # This special method is used to assign to the SDS dataset
        # using "extended slice syntax". The extended slice syntax
        # is a perfect match for the "start", "count" and "stride"
        # arguments to the SDwritedata() function, and is much more easy
        # to use.

        # Compute arguments to 'SDwritedata_0()'.
        start, count, stride = self.__buildStartCountStride(elem)
        # A sequence type is needed. Convert a single number to a list.
        if type(data) in [int, float]:
            data = [data]
        # Assign.
        self.set(data, start, count, stride)

    def endaccess(self):
        """Terminates access to the SDS.

        Args::

          no argument

        Returns::

          None.

        The SDS instance should not be used afterwards.
        The 'endaccess()' method is implicitly called when
        the SDS instance is deleted.

        C library equivalent : SDendaccess
                                                 """

        status = _C.SDendaccess(self._id)
        _checkErr('endaccess', status, "cannot execute")
        self._id = None    # Invalidate identifier


    def dim(self, dim_index):
        """Get an SDim instance given a dimension index number.

        Args::

          dim_index index number of the dimension (numbering starts at 0)

        C library equivalent : SDgetdimid
                                                    """
        id = _C.SDgetdimid(self._id, dim_index)
        _checkErr('dim', id, 'invalid SDS identifier or dimension index')
        return SDim(self, id, dim_index)

    def get(self, start=None, count=None, stride=None):
        """Read data from the dataset.

        Args::

          start   : indices where to start reading in the data array;
                    default to 0 on all dimensions
          count   : number of values to read along each dimension;
                    default to the current length of all dimensions
          stride  : sampling interval along each dimension;
                    default to 1 on all dimensions

          For n-dimensional datasets, those 3 parameters are entered
          using lists. For one-dimensional datasets, integers
          can also be used.

          Note that, to read the whole dataset contents, one should
          simply call the method with no argument.

        Returns::

          numpy array initialized with the data.

        C library equivalent : SDreaddata

        The dataset can also be read using the familiar indexing and
        slicing notation, like ordinary python sequences.
        See "High level variable access".

                                                       """

        # Obtain SDS info.
        try:
            sds_name, rank, dim_sizes, data_type, n_attrs = self.info()
            if isinstance(dim_sizes, type(1)):
                dim_sizes = [dim_sizes]
        except HDF4Error:
            raise HDF4Error('get : cannot execute')

        # Validate args.
        if start is None:
            start = [0] * rank
        elif isinstance(start, type(1)):
            start = [start]
        if count is None:
            count = dim_sizes
            if count[0] == 0:
                count[0] = 1
        elif isinstance(count, type(1)):
            count = [count]
        if stride is None:
            stride = [1] * rank
        elif isinstance(stride, type(1)):
            stride = [stride]
        if len(start) != rank or len(count) != rank or len(stride) != rank:
            raise HDF4Error('get : start, stride or count ' \
                             'do not match SDS rank')
        for n in range(rank):
            if start[n] < 0 or start[n] + \
                  (abs(count[n]) - 1) * stride[n] >= dim_sizes[n]:
                raise HDF4Error('get arguments violate ' \
                                 'the size (%d) of dimension %d' \
                                 % (dim_sizes[n], n))
        if not data_type in SDC.equivNumericTypes:
            raise HDF4Error('get cannot currrently deal with '\
                             'the SDS data type')

        return _C._SDreaddata_0(self._id, data_type, start, count, stride)

    def set(self, data, start=None, count=None, stride=None):
        """Write data to the dataset.

        Args::

          data    : array of data to write; can be given as a numpy
                    array, or as Python sequence (whose elements can be
                    imbricated sequences)
          start   : indices where to start writing in the dataset;
                    default to 0 on all dimensions
          count   : number of values to write along each dimension;
                    default to the current length of dataset dimensions
          stride  : sampling interval along each dimension;
                    default to 1 on all dimensions

          For n-dimensional datasets, those 3 parameters are entered
          using lists. For one-dimensional datasets, integers
          can also be used.

          Note that, to write the whole dataset at once, one has simply
          to call the method with the dataset values in parameter
          'data', omitting all other parameters.

        Returns::

          None.

        C library equivalent : SDwritedata

        The dataset can also be written using the familiar indexing and
        slicing notation, like ordinary python sequences.
        See "High level variable access".

                                              """


        # Obtain SDS info.
        try:
            sds_name, rank, dim_sizes, data_type, n_attrs = self.info()
            if isinstance(dim_sizes, type(1)):
                dim_sizes = [dim_sizes]
        except HDF4Error:
            raise HDF4Error('set : cannot execute')

        # Validate args.
        if start is None:
            start = [0] * rank
        elif isinstance(start, type(1)):
            start = [start]
        if count is None:
            count = dim_sizes
            if count[0] == 0:
                count[0] = 1
        elif isinstance(count, type(1)):
            count = [count]
        if stride is None:
            stride = [1] * rank
        elif isinstance(stride, type(1)):
            stride = [stride]
        if len(start) != rank or len(count) != rank or len(stride) != rank:
            raise HDF4Error('set : start, stride or count '\
                             'do not match SDS rank')
        unlimited = self.isrecord()
        for n in range(rank):
            ok = 1
            if start[n] < 0:
                ok = 0
            elif n > 0 or not unlimited:
                if start[n] + (abs(count[n]) - 1) * stride[n] >= dim_sizes[n]:
                    ok = 0
            if not ok:
                raise HDF4Error('set arguments violate '\
                                 'the size (%d) of dimension %d' \
                                 % (dim_sizes[n], n))
        # ??? Check support for UINT16
        if not data_type in SDC.equivNumericTypes:
            raise HDF4Error('set cannot currrently deal '\
                             'with the SDS data type')

        _C._SDwritedata_0(self._id, data_type, start, count, data, stride)

    def __buildStartCountStride(self, elem):

        # Create the 'start', 'count', 'slice' and 'stride' tuples that
        # will be passed to '_SDreaddata_0'/'_SDwritedata_0'.
        #   start     starting indices along each dimension
        #   count     count of values along each dimension; a value of -1
        #             indicates that and index, not a slice, was applied to
        #             the dimension; in that case, the dimension should be
        #             dropped from the output array.
        #   stride    strides along each dimension


        # Make sure the indexing expression does not exceed the variable
        # number of dimensions.
        dsName, nDims, shape, dsType, nAttr = self.info()
        if isinstance(elem, tuple):
            if len(elem) > nDims:
                raise HDF4Error("get", 0,
                               "indexing expression exceeds variable "
                               "number of dimensions")
        else:   # Convert single index to sequence
            elem = [elem]
        if isinstance(shape, int):
            shape = [shape]

        start = []
        count = []
        stride = []
        n = -1
        unlimited = self.isrecord()
        for e in elem:
            n += 1
            # See if the dimension is unlimited (always at index 0)
            unlim = n == 0 and unlimited
            # Simple index
            if isinstance(e, int):
                isslice = False
                if e < 0 :
                    e += shape[n]
                # Respect standard python list behavior: it is illegal to
                # specify an out of bound index (except for the
                # unlimited dimension).
                if e < 0 or (not unlim and e >= shape[n]):
                    raise IndexError("index out of range")
                beg = e
                end = e + 1
                inc = 1
            # Slice index. Respect Python syntax for slice upper bounds,
            # which are not included in the resulting slice. Also, if the
            # upper bound exceed the dimension size, truncate it.
            elif isinstance(e, slice):
                isslice = True
                # None or 0 means not specified
                if e.start:
                    beg = e.start
                    if beg < 0:
                        beg += shape[n]
                else:
                    beg = 0
                # None of maxint means not specified
                if e.stop and e.stop != sys.maxsize:
                    end = e.stop
                    if end < 0:
                        end += shape[n]
                else:
                    end = shape[n]
                # None means not specified
                if e.step:
                    inc = e.step
                else:
                    inc = 1
            # Bug
            else:
                raise ValueError("Bug: unexpected element type to __getitem__")

            # Clip end index (except if unlimited dimension)
            # and compute number of elements to get.
            if not unlim and end > shape[n]:
                end = shape[n]
            if isslice:
                cnt = (end - beg) // inc
                if cnt * inc < end - beg:
                    cnt += 1
            else:
                cnt = -1
            start.append(beg)
            count.append(cnt)
            stride.append(inc)

        # Complete missing dimensions
        while n < nDims - 1:
            n += 1
            start.append(0)
            count.append(shape[n])
            stride.append(1)

        # Done
        return start, count, stride

    def info(self):
        """Retrieves information about the dataset.

        Args::

          no argument

        Returns::

          5-element tuple holding:

          - dataset name
          - dataset rank (number of dimensions)
          - dataset shape, that is a list giving the length of each
            dataset dimension; if the first dimension is unlimited, then
            the first value of the list gives the current length of the
            unlimited dimension
          - data type (one of the SDC.xxx values)
          - number of attributes defined for the dataset

        C library equivalent : SDgetinfo
                                                       """

        buf = _C.array_int32(_C.H4_MAX_VAR_DIMS)
        status, sds_name, rank, data_type, n_attrs = \
                _C.SDgetinfo(self._id, buf)
        _checkErr('info', status, "cannot execute")
        dim_sizes = _array_to_ret(buf, rank)
        return sds_name, rank, dim_sizes, data_type, n_attrs

    def checkempty(self):
        """Determine whether the dataset is empty.

        Args::

          no argument

        Returns::

          True(1) if dataset is empty, False(0) if not

        C library equivalent : SDcheckempty
                                                 """

        status, emptySDS = _C.SDcheckempty(self._id)
        _checkErr('checkempty', status, 'invalid SDS identifier')
        return emptySDS

    def ref(self):
        """Get the reference number of the dataset.

        Args::

          no argument

        Returns::

          dataset reference number

        C library equivalent : SDidtoref
                                              """

        sds_ref = _C.SDidtoref(self._id)
        _checkErr('idtoref', sds_ref, 'illegal SDS identifier')
        return sds_ref

    def iscoordvar(self):
        """Determine whether the dataset is a coordinate variable
        (holds a dimension scale). A coordinate variable is created
        when a dimension is assigned a set of scale values.

        Args::

          no argument

        Returns::

          True(1) if the dataset represents a coordinate variable,
          False(0) if not

        C library equivalent : SDiscoordvar
                                           """

        return _C.SDiscoordvar(self._id)   # no error status here

    def isrecord(self):
        """Determines whether the dataset is appendable
        (contains an unlimited dimension). Note that if true, then
        the unlimited dimension is always dimension number 0.

        Args::

          no argument

        Returns::

          True(1) if the dataset is appendable, False(0) if not.

        C library equivalent : SDisrecord
                                        """

        return _C.SDisrecord(self._id)     # no error status here


    def getcal(self):
        """Retrieve the SDS calibration coefficients.

        Args::

          no argument

        Returns::

          5-element tuple holding:

          - cal: calibration factor (attribute 'scale_factor')
          - cal_error : calibration factor error
                        (attribute 'scale_factor_err')
          - offset: calibration offset (attribute 'add_offset')
          - offset_err : offset error (attribute 'add_offset_err')
          - data_type : type of the data resulting from applying
                        the calibration formula to the dataset values
                        (attribute 'calibrated_nt')

        An exception is raised if no calibration data are defined.

        Original dataset values 'orival' are converted to calibrated
        values 'calval' through the formula::

           calval = cal * (orival - offset)

        The calibration coefficients are part of the so-called
        "standard" SDS attributes. The values inside the tuple returned
        by 'getcal' are those of the following attributes, in order::

          scale_factor, scale_factor_err, add_offset, add_offset_err,
          calibrated_nt

        C library equivalent: SDgetcal()
                                               """

        status, cal, cal_error, offset, offset_err, data_type = \
                         _C.SDgetcal(self._id)
        _checkErr('getcal', status, 'no calibration record')
        return cal, cal_error, offset, offset_err, data_type

    def getdatastrs(self):
        """Retrieve the dataset standard string attributes.

        Args::

          no argument

        Returns::

          4-element tuple holding:

          - dataset label string (attribute 'long_name')
          - dataset unit (attribute 'units')
          - dataset output format (attribute 'format')
          - dataset coordinate system (attribute 'coordsys')

        The values returned by 'getdatastrs' are part of the
        so-called "standard" SDS attributes.  Those 4 values
        correspond respectively to the following attributes::

          long_name, units, format, coordsys .

        C library equivalent: SDgetdatastrs
                                                       """

        status, label, unit, format, coord_system = \
               _C.SDgetdatastrs(self._id, 128)
        _checkErr('getdatastrs', status, 'cannot execute')
        return label, unit, format, coord_system

    def getfillvalue(self):
        """Retrieve the dataset fill value.

        Args::

          no argument

        Returns::

          dataset fill value (attribute '_FillValue')

        An exception is raised if the fill value is not set.

        The fill value is part of the so-called "standard" SDS
        attributes, and corresponds to the following attribute::

          _FillValue

        C library equivalent: SDgetfillvalue
                                                   """

        # Obtain SDS data type.
        try:
            sds_name, rank, dim_sizes, data_type, n_attrs = \
                                self.info()
        except HDF4Error:
            raise HDF4Error('getfillvalue : invalid SDS identifier')
        n_values = 1   # Fill value stands for 1 value.

        convert = _array_to_ret
        if data_type == SDC.CHAR8:
            buf = _C.array_byte(n_values)
            convert = _array_to_str

        elif data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            buf = _C.array_int8(n_values)

        elif data_type == SDC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("getfillvalue: SDS has an illegal type or " \
                             "unsupported type %d" % data_type)

        status = _C.SDgetfillvalue(self._id, buf)
        _checkErr('getfillvalue', status, 'fill value not set')
        return convert(buf, n_values)

    def getrange(self):
        """Retrieve the dataset min and max values.

        Args::

          no argument

        Returns::

          (min, max) tuple (attribute 'valid_range')

          Note that those are the values as stored
          by the 'setrange' method. 'getrange' does *NOT* compute the
          min and max from the current dataset contents.

        An exception is raised if the range is not set.

        The range returned by 'getrange' is part of the so-called
        "standard" SDS attributes. It corresponds to the following
        attribute::

          valid_range

        C library equivalent: SDgetrange
                                                       """

        # Obtain SDS data type.
        try:
            sds_name, rank, dim_sizes, data_type, n_attrs = \
                               self.info()
        except HDF4Error:
            raise HDF4Error('getrange : invalid SDS identifier')
        n_values = 1

        convert = _array_to_ret
        if data_type == SDC.CHAR8:
            buf1 = _C.array_byte(n_values)
            buf2 = _C.array_byte(n_values)
            convert = _array_to_str

        elif data_type in  [SDC.UCHAR8, SDC.UINT8]:
            buf1 = _C.array_byte(n_values)
            buf2 = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            buf1 = _C.array_int8(n_values)
            buf2 = _C.array_int8(n_values)

        elif data_type == SDC.INT16:
            buf1 = _C.array_int16(n_values)
            buf2 = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf1 = _C.array_uint16(n_values)
            buf2 = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf1 = _C.array_int32(n_values)
            buf2 = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf1 = _C.array_uint32(n_values)
            buf2 = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf1 = _C.array_float32(n_values)
            buf2 = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf1 = _C.array_float64(n_values)
            buf2 = _C.array_float64(n_values)

        else:
            raise HDF4Error("getrange: SDS has an illegal or " \
                             "unsupported type %d" % data)

        # Note: The C routine returns the max in buf1 and the min
        # in buf2. We swap the values returned by the Python
        # interface, since it is more natural to return
        # min first, then max.
        status = _C.SDgetrange(self._id, buf1, buf2)
        _checkErr('getrange', status, 'range not set')
        return convert(buf2, n_values), convert(buf1, n_values)

    def setcal(self, cal, cal_error, offset, offset_err, data_type):
        """Set the dataset calibration coefficients.

        Args::

          cal         the calibraton factor (attribute 'scale_factor')
          cal_error   calibration factor error
                      (attribute 'scale_factor_err')
          offset      offset value (attribute 'add_offset')
          offset_err  offset error (attribute 'add_offset_err')
          data_type   data type of the values resulting from applying the
                      calibration formula to the dataset values
                      (one of the SDC.xxx constants)
                      (attribute 'calibrated_nt')

        Returns::

          None

        See method 'getcal' for the definition of the calibration
        formula.

        Calibration coefficients are part of the so-called standard
        SDS attributes. Calling 'setcal' is equivalent to setting
        the following attributes, which correspond to the method
        parameters, in order::

          scale_factor, scale_factor_err, add_offset, add_offset_err,
          calibrated_nt

        C library equivalent: SDsetcal
                                                      """

        status = _C.SDsetcal(self._id, cal, cal_error,
                             offset, offset_err, data_type)
        _checkErr('setcal', status, 'cannot execute')

    def setdatastrs(self, label, unit, format, coord_sys):
        """Set the dataset standard string type attributes.

        Args::

          label         dataset label (attribute 'long_name')
          unit          dataset unit (attribute 'units')
          format        dataset format (attribute 'format')
          coord_sys     dataset coordinate system (attribute 'coordsys')

        Returns::

          None

        Those strings are part of the so-called standard
        SDS attributes. Calling 'setdatastrs' is equivalent to setting
        the following attributes, which correspond to the method
        parameters, in order::

          long_name, units, format, coordsys

        C library equivalent: SDsetdatastrs
                                                     """

        status = _C.SDsetdatastrs(self._id, label, unit, format, coord_sys)
        _checkErr('setdatastrs', status, 'cannot execute')

    def setfillvalue(self, fill_val):
        """Set the dataset fill value.

        Args::

          fill_val   dataset fill value (attribute '_FillValue')

        Returns::

          None

        The fill value is part of the so-called "standard" SDS
        attributes. Calling 'setfillvalue' is equivalent to setting
        the following attribute::

          _FillValue

        C library equivalent: SDsetfillvalue
                                                           """

        # Obtain SDS data type.
        try:
            sds_name, rank, dim_sizes, data_type, n_attrs = self.info()
        except HDF4Error:
            raise HDF4Error('setfillvalue : cannot execute')
        n_values = 1   # Fill value stands for 1 value.

        if data_type == SDC.CHAR8:
            buf = _C.array_byte(n_values)

        elif data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            # SWIG refuses negative values here. We found that if we
            # pass them as byte values, it will work.
            buf = _C.array_int8(n_values)
            if fill_val >= 0:
                fill_val &= 0x7f
            else:
                fill_val = abs(fill_val) & 0x7f
                if fill_val:
                    fill_val = 256 - fill_val
                else:
                    fill_val = 128    # -128 in 2's complement

        elif data_type == SDC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("setfillvalue: SDS has an illegal or " \
                             "unsupported type %d" % data_type)

        buf[0] = fill_val
        status = _C.SDsetfillvalue(self._id, buf)
        _checkErr('setfillvalue', status, 'cannot execute')


    def setrange(self, min, max):
        """Set the dataset min and max values.

        Args::

          min        dataset minimum value (attribute 'valid_range')
          max        dataset maximum value (attribute 'valid_range')


        Returns::

          None

        The data range is part of the so-called "standard" SDS
        attributes. Calling method 'setrange' is equivalent to
        setting the following attribute with a 2-element [min,max]
        array::

          valid_range


        C library equivalent: SDsetrange
                                                   """

        # Obtain SDS data type.
        try:
            sds_name, rank, dim_sizes, data_type, n_attrs = self.info()
        except HDF4Error:
            raise HDF4Error('setrange : cannot execute')
        n_values = 1

        if data_type == SDC.CHAR8:
            buf1 = _C.array_byte(n_values)
            buf2 = _C.array_byte(n_values)

        elif data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf1 = _C.array_byte(n_values)
            buf2 = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            # SWIG refuses negative values here. We found that if we
            # pass them as byte values, it will work.
            buf1 = _C.array_int8(n_values)
            buf2 = _C.array_int8(n_values)
            v = min
            if v >= 0:
                v &= 0x7f
            else:
                v = abs(v) & 0x7f
                if v:
                    v = 256 - v
                else:
                    v = 128    # -128 in 2's complement
            min = v
            v = max
            if v >= 0:
                v &= 0x7f
            else:
                v = abs(v) & 0x7f
                if v:
                    v = 256 - v
                else:
                    v = 128    # -128 in 2's complement
            max = v

        elif data_type == SDC.INT16:
            buf1 = _C.array_int16(n_values)
            buf2 = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf1 = _C.array_uint16(n_values)
            buf2 = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf1 = _C.array_int32(n_values)
            buf2 = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf1 = _C.array_uint32(n_values)
            buf2 = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf1 = _C.array_float32(n_values)
            buf2 = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf1 = _C.array_float64(n_values)
            buf2 = _C.array_float64(n_values)

        else:
            raise HDF4Error("SDsetrange: SDS has an illegal or " \
                             "unsupported type %d" % data_type)

        buf1[0] = max
        buf2[0] = min
        status = _C.SDsetrange(self._id, buf1, buf2)
        _checkErr('setrange', status, 'cannot execute')

    def getcompress(self):
        """Retrieves info about dataset compression type and mode.

        Args::

          no argument

        Returns::

          tuple holding:

          - compression type (one of the SDC.COMP_xxx constants)
          - optional values, depending on the compression type
              COMP_NONE       0 value    no additional value
              COMP_SKPHUFF    1 value  : skip size
              COMP_DEFLATE    1 value  : gzip compression level (1 to 9)
              COMP_SZIP       5 values : options mask,
                                         pixels per block (2 to 32)
                                         pixels per scanline,
                                         bits per pixel (number of bits in the SDS datatype)
                                         pixels (number of elements in the SDS)

                                         Note: in the context of an SDS, the word "pixel"
                                         should really be understood as meaning "data element",
                                         eg a cell value inside a multidimensional grid.
                                         Test the options mask against constants SDC.COMP_SZIP_NN
                                         and SDC.COMP_SZIP_EC, eg :
                                           if optionMask & SDC.COMP_SZIP_EC:
                                               print "EC encoding scheme used"

        An exception is raised if dataset is not compressed.

        .. note::
            Starting with v0.8, an exception is always raised if
            pyhdf was installed with the NOCOMPRESS macro set.

        C library equivalent: SDgetcompress
                                                           """

        status, comp_type, value, v2, v3, v4, v5 = _C._SDgetcompress(self._id)
        _checkErr('getcompress', status, 'no compression')
        if comp_type == SDC.COMP_NONE:
            return (comp_type,)
        elif comp_type == SDC.COMP_SZIP:
            return comp_type, value, v2, v3, v4, v5
        else:
            return comp_type, value

    def setcompress(self, comp_type, value=0, v2=0):
        """Compresses the dataset using a specified compression method.

        Args::

          comp_type    compression type, identified by one of the
                       SDC.COMP_xxx constants
          value,v2     auxiliary value(s) needed by some compression types
                         SDC.COMP_SKPHUFF   Skipping-Huffman; compression value=data size in bytes, v2 is ignored
                         SDC.COMP_DEFLATE   Gzip compression; value=deflate level (1 to 9), v2 is ignored
                         SDC.COMP_SZIP      Szip compression; value=encoding scheme (SDC.COMP_SZIP_EC or
                                            SDC.COMP_SZIP_NN), v2=pixels per block (2 to 32)

        Returns::

            None

        .. note::
             Starting with v0.8, an exception is always raised if
             pyhdf was installed with the NOCOMPRESS macro set.

        SDC.COMP_DEFLATE applies the GZIP compression to the dataset,
        and the value varies from 1 to 9, according to the level of
        compression desired.

        SDC.COMP_SZIP compresses the dataset using the SZIP algorithm. See the HDF User's Guide
        for details about the encoding scheme and the number of pixels per block. SZIP is new
        with HDF 4.2.

        'setcompress' must be called before writing to the dataset.
        The dataset must be written all at once, unless it is
        appendable (has an unlimited dimension). Updating the dataset
        in not allowed. Refer to the HDF user's guide for more details
        on how to use data compression.

        C library equivalent: SDsetcompress
                                                          """

        status = _C._SDsetcompress(self._id, comp_type, value, v2)
        _checkErr('setcompress', status, 'cannot execute')


    def setexternalfile(self, filename, offset=0):
        """Store the dataset data in an external file.

        Args::

          filename    external file name
          offset      offset in bytes where to start writing in
                      the external file

        Returns::

            None

        C library equivalent : SDsetexternalfile
                                                  """

        status = _C.SDsetexternalfile(self._id, filename, offset)
        _checkErr('setexternalfile', status, 'execution error')

    def attr(self, name_or_index):
        """Create an SDAttr instance representing an SDS
        (dataset) attribute.

        Args::

          name_or_index   attribute name or index number; if a name is
                          given, the attribute may not exist

        Returns::

          SDAttr instance for the attribute. Call the methods of this
          class to query, read or set the attribute.

        C library equivalent : no equivalent

                                """

        return SDAttr(self, name_or_index)

    def attributes(self, full=0):
        """Return a dictionnary describing every attribute defined
        on the dataset.

        Args::

          full      true to get complete info about each attribute
                    false to report only each attribute value

        Returns::

          Empty dictionnary if no attribute defined.
          Otherwise, dictionnary where each key is the name of a
          dataset attribute. If parameter 'full' is false,
          key value is the attribute value. If 'full' is true,
          key value is a tuple with the following elements:

          - attribute value
          - attribute index number
          - attribute type
          - attribute length

        C library equivalent : no equivalent
                                                    """

        # Get the number of dataset attributes.
        natts = self.info()[4]

        # Inquire each attribute
        res = {}
        for n in range(natts):
            a = self.attr(n)
            name, aType, nVal = a.info()
            if full:
                res[name] = (a.get(), a.index(), aType, nVal)
            else:
                res[name] = a.get()

        return res

    def dimensions(self, full=0):
        """Return a dictionnary describing every dataset dimension.

        Args::

          full      true to get complete info about each dimension
                    false to report only each dimension length

        Returns::

          Dictionnary where each key is a dimension name. If no name
          has been given to the dimension, the key is set to
          'fakeDimx' where 'x' is the dimension index number.
          If parameter 'full' is false, key value is the dimension
          length. If 'full' is true, key value is a 5-element tuple
          with the following elements:

          - dimension length; for an unlimited dimension, the reported
            length is the current dimension length
          - dimension index number
          - 1 if the dimension is unlimited, 0 otherwise
          - dimension scale type, or 0 if no scale is defined for
            the dimension
          - number of attributes defined on the dimension

        C library equivalent : no equivalent
                                                    """

        # Get the number of dimensions and their lengths.
        nDims, dimLen = self.info()[1:3]
        if isinstance(dimLen, int):    # need a sequence
            dimLen = [dimLen]
        # Check if the dataset is appendable.
        unlim = self.isrecord()

        # Inquire each dimension
        res = {}
        for n in range(nDims):
            d = self.dim(n)
            # The length reported by info() is 0 for an unlimited dimension.
            # Rather use the lengths reported by SDS.info()
            name, k, scaleType, nAtt = d.info()
            length = dimLen[n]
            if full:
                res[name] = (length, n, unlim and n == 0,
                             scaleType, nAtt)
            else:
                res[name] = length

        return res


class SDim(object):
    """The SDim class implements a dimension object.
       There can be one dimension object for each dataset dimension.
       To create an SDim instance, call the dim() method of an SDS class
       instance. To set attributes on an SDim instance, call the
       SDim.attr() method to create an attribute instance, then call the
       methods of this instance.  Attributes can also be set using the
       "dot notation". """

    def __init__(self, sds, id, index):
        """Init an SDim instance. This method should not be called
        directly by the user program. To create an SDim instance,
        call the SDS.dim() method.
                                                 """

        # Args:
        #  sds    SDS instance
        #  id     dimension identifier
        #  index  index number of the dimension

        # SDim private attributes
        #  _sds    sds instance
        #  _id     dimension identifier
        #  _index  dimension index number

        self._sds = sds
        self._id = id
        self._index = index

    def __getattr__(self, name):
        # Get value(s) of SDim attribute 'name'.

        return _getattr(self, name)

    def __setattr__(self, name, value):
        # Set value(s) of SDim attribute 'name'.

        _setattr(self, name, value, ['_sds', '_id', '_index'])


    def info(self):
        """Return info about the dimension instance.

        Args::

          no argument

        Returns::

          4-element tuple holding:

          - dimension name; 'fakeDimx' is returned if the dimension
            has not been named yet, where 'x' is the dimension
            index number
          - dimension length; 0 is returned if the dimension is unlimited;
            call the SDim.length() or SDS.info() methods to obtain the
            current dimension length
          - scale data type (one of the SDC.xxx constants); 0 is
            returned if no scale has been set on the dimension
          - number of attributes attached to the dimension

        C library equivalent : SDdiminfo
                                                    """
        status, dim_name, dim_size, data_type, n_attrs = \
                _C.SDdiminfo(self._id)
        _checkErr('info', status, 'cannot execute')
        return dim_name, dim_size, data_type, n_attrs

    def length(self):
        """Return the dimension length. This method is usefull
        to quickly retrieve the current length of an unlimited
        dimension.

        Args::

          no argument

        Returns::

          dimension length; if the dimension is unlimited, the
          returned value is the current dimension length

        C library equivalent : no equivalent
                                                   """

        return self._sds.info()[2][self._index]

    def setname(self, dim_name):
        """Set the dimension name.

        Args::

          dim_name    dimension name; setting 2 dimensions to the same
                      name make the dimensions "shared"; in order to be
                      shared, the dimesions must be deined similarly.

        Returns::

          None

        C library equivalent : SDsetdimname
                                                            """

        status = _C.SDsetdimname(self._id, dim_name)
        _checkErr('setname', status, 'cannot execute')


    def getscale(self):
        """Obtain the scale values along a dimension.

        Args::

          no argument

        Returns::

          list with the scale values; the list length is equal to the
          dimension length; the element type is equal to the dimension
          data type, as set when the 'setdimscale()' method was called.

        C library equivalent : SDgetdimscale
                                                  """

        # Get dimension info. If data_type is 0, no scale have been set
        # on the dimension.
        status, dim_name, dim_size, data_type, n_attrs = _C.SDdiminfo(self._id)
        _checkErr('getscale', status, 'cannot execute')
        if data_type == 0:
            raise HDF4Error("no scale set on that dimension")

        # dim_size is 0 for an unlimited dimension. The actual length is
        # obtained through SDgetinfo.
        if dim_size == 0:
            dim_size = self._sds.info()[2][self._index]

        # Get scale values.
        if data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf = _C.array_byte(dim_size)

        elif data_type == SDC.INT8:
            buf = _C.array_int8(dim_size)

        elif data_type == SDC.INT16:
            buf = _C.array_int16(dim_size)

        elif data_type == SDC.UINT16:
            buf = _C.array_uint16(dim_size)

        elif data_type == SDC.INT32:
            buf = _C.array_int32(dim_size)

        elif data_type == SDC.UINT32:
            buf = _C.array_uint32(dim_size)

        elif data_type == SDC.FLOAT32:
            buf = _C.array_float32(dim_size)

        elif data_type == SDC.FLOAT64:
            buf = _C.array_float64(dim_size)

        else:
            raise HDF4Error("getscale: dimension has an "\
                             "illegal or unsupported type %d" % data_type)

        status = _C.SDgetdimscale(self._id, buf)
        _checkErr('getscale', status, 'cannot execute')
        return _array_to_ret(buf, dim_size)

    def setscale(self, data_type, scale):
        """Initialize the scale values along the dimension.

        Args::

          data_type    data type code (one of the SDC.xxx constants)
          scale        sequence holding the scale values; the number of
                       values must match the current length of the dataset
                       along that dimension

        C library equivalent : SDsetdimscale

        Setting a scale on a dimension generates what HDF calls a
        "coordinate variable". This is a rank 1 dataset similar to any
        other dataset, which is created to hold the scale values. The
        dataset name is identical to that of the dimension on which
        setscale() is called, and the data type passed in 'data_type'
        determines the type of the dataset. To distinguish between such
        a dataset and a "normal" dataset, call the iscoordvar() method
        of the dataset instance.
                                                         """

        try:
            n_values = len(scale)
        except:
            n_values = 1

        # Validate args
        info = self._sds.info()
        if info[1] == 1:
            dim_size = info[2]
        else:
            dim_size = info[2][self._index]
        if n_values != dim_size:
            raise HDF4Error('number of scale values (%d) does not match ' \
                             'dimension size (%d)' % (n_values, dim_size))

        if data_type == SDC.CHAR8:
            buf = _C.array_byte(n_values)
            # Allow a string as the scale argument.
            # Becomes a noop if already a list.
            scale = list(scale)
            for n in range(n_values):
                scale[n] = ord(scale[n])

        elif data_type in [SDC.UCHAR8, SDC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == SDC.INT8:
            # SWIG refuses negative values here. We found that if we
            # pass them as byte values, it will work.
            buf = _C.array_int8(n_values)
            scale = list(scale)
            for n in range(n_values):
                v = scale[n]
                if v >= 0:
                    v &= 0x7f
                else:
                    v = abs(v) & 0x7f
                    if v:
                        v = 256 - v
                    else:
                        v = 128         # -128 in 2's complement
                scale[n] = v

        elif data_type == SDC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == SDC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == SDC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == SDC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == SDC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == SDC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("setscale: illegal or usupported data_type")

        if n_values == 1:
            buf[0] = scale
        else:
            for n in range(n_values):
                buf[n] = scale[n]
        status = _C.SDsetdimscale(self._id, n_values, data_type, buf)
        _checkErr('setscale', status, 'cannot execute')

    def getstrs(self):
        """Retrieve the dimension standard string attributes.

        Args::

          no argument

        Returns::

          3-element tuple holding:
            -dimension label  (attribute 'long_name')
            -dimension unit   (attribute 'units')
            -dimension format (attribute 'format')

        An exception is raised if the standard attributes have
        not been set.

        C library equivalent: SDgetdimstrs
                                                """

        status, label, unit, format = _C.SDgetdimstrs(self._id, 128)
        _checkErr('getstrs', status, 'cannot execute')
        return label, unit, format

    def setstrs(self, label, unit, format):
        """Set the dimension standard string attributes.

        Args::

          label   dimension label  (attribute 'long_name')
          unit    dimension unit   (attribute 'units')
          format  dimension format (attribute 'format')

        Returns::

          None

        C library equivalent: SDsetdimstrs
                                                     """

        status = _C.SDsetdimstrs(self._id, label, unit, format)
        _checkErr('setstrs', status, 'cannot execute')

    def attr(self, name_or_index):
        """Create an SDAttr instance representing an SDim
        (dimension) attribute.

        Args::

          name_or_index   attribute name or index number; if a name is
                          given, the attribute may not exist; in that
                          case, the attribute is created when the
                          instance set() method is called

        Returns::

          SDAttr instance for the attribute. Call the methods of this
          class to query, read or set the attribute.

        C library equivalent : no equivalent

                                """

        return SDAttr(self, name_or_index)

    def attributes(self, full=0):
        """Return a dictionnary describing every attribute defined
        on the dimension.

        Args::

          full      true to get complete info about each attribute
                    false to report only each attribute value

        Returns::

          Empty dictionnary if no attribute defined.
          Otherwise, dictionnary where each key is the name of a
          dimension attribute. If parameter 'full' is false,
          key value is the attribute value. If 'full' is true,
          key value is a tuple with the following elements:

          - attribute value
          - attribute index number
          - attribute type
          - attribute length

        C library equivalent : no equivalent
                                                    """

        # Get the number of dataset attributes.
        natts = self.info()[3]

        # Inquire each attribute
        res = {}
        for n in range(natts):
            a = self.attr(n)
            name, aType, nVal = a.info()
            if full:
                res[name] = (a.get(), a.index(), aType, nVal)
            else:
                res[name] = a.get()

        return res



###########################
# Support functions
###########################

def _getattr(obj, name):
    # Called by the __getattr__ method of the SD, SDS and SDim objects.

    # Python will call __getattr__ to see if the class wants to
    # define certain missing methods (__str__, __len__, etc).
    # Always fail if the name starts with two underscores.
    if name[:2] == '__':
        raise AttributeError
    # See if we deal with an SD attribute.
    a = SDAttr(obj, name)
    try:
        index = a.index()
    except HDF4Error:
        raise AttributeError("attribute not found")
    # Return attribute value(s).
    return a.get()

def _setattr(obj, name, value, privAttr):
    # Called by the __setattr__ method of the SD, SDS and SDim objects.

    # Be careful with private attributes.
    #if name in privAttr:
    if name[0] == '_':
        obj.__dict__[name] = value
        return

    # Treat everything else as an HDF attribute.
    if type(value) not in [list, tuple]:
        value = [value]
    typeList = []
    for v in value:
        t = type(v)
        # Prohibit mixing numeric types and strings.
        if t in [int, float] and \
               not str in typeList:
            if t not in typeList:
                typeList.append(t)
        # Prohibit sequence of strings or a mix of numbers and string.
        elif t == str and not typeList:
            typeList.append(t)
        else:
            typeList = []
            break
    if str in typeList:
        xtype = SDC.CHAR8
        value = value[0]
    # double is "stronger" than int
    elif float in typeList:
        xtype = SDC.FLOAT64
    elif int in typeList:
        xtype = SDC.INT32
    else:
        raise HDF4Error("Illegal attribute value")

    # Assign value
    try:
        a = SDAttr(obj, name)
        a.set(xtype, value)
    except HDF4Error as msg:
        raise HDF4Error("cannot set attribute: %s" % msg)

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
    return ''.join(chrs)
