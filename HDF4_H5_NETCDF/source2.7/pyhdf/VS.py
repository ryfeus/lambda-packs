# $Id: VS.py,v 1.4 2005-07-14 01:36:41 gosselin_a Exp $
# $Log: not supported by cvs2svn $
# Revision 1.3  2004/08/02 17:06:20  gosselin
# pyhdf-0.7.2
#
# Revision 1.2  2004/08/02 15:36:04  gosselin
# pyhdf-0.7-1
#
# Author: Andre Gosselin
#         Maurice-Lamontagne Institute
#         gosselina@dfo-mpo.gc.ca

"""
VS (Vdata table) API (:mod:`pyhdf.VS`)
======================================

A module of the pyhdf package implementing the VS (Vdata table)
API of the NCSA HDF4 library.
(see: hdf.ncsa.uiuc.edu)

Introduction
------------
VS is one of the modules composing pyhdf, a python package implementing
the NCSA HDF library and letting one manage HDF files from within a python
program. Two versions of the HDF library currently exist, version 4 and
version 5. pyhdf only implements version 4 of the library. Many
different APIs are to be found inside the HDF4 specification.
Currently, pyhdf implements just a few of those: the SD, VS and V APIs.
Other APIs should be added in the future (GR, AN, etc).

VS allows the definition of structured data tables inside an HDF file.
Those tables are designated as "vdatas" (the name has to do with data
associated with the "vertices" of geometrical models, the storage of which
the API was originally designed for). A vdata is composed of a fixed
number of columns (also called fields), where a column can store a fixed
number of data values, all of the same type. The number of values allowed
inside a field is called the "order" of the field. A table is composed of a
varying number of rows (also called records), a record representing the
sequence of values stored in each field of the vdata.

A vdata is associated with a descriptive name, and likewise each field of
the vdata. A vdata can also be tagged with a "class" to further describe the
vdata purpose. Records and fields are identified by a zero-based index.
An arbitrary number of attributes of different types can be attached to
a vdata as a whole, or to its individual fields. An attribute is a
(name, value) pair, where "value" can be of many types, and be either
single or multi-valued. The number of values stored in an attribute is
called the "order" of the attribute.

The following example illustrates a simple vdata that could be stored
inside an HDF file. See section "Programming models" for an example
program implementing this vdata.

                             INVENTORY (experimental status)

            ======     ===========     ===   ========  ========
            partid     description     qty   wght(lb)  price($)
            ======     ===========     ===   ========  ========
            Q1234       bolt           12     0.01      0.05
            B5432       brush          10     0.4       4.25
            S7613       scissor         2     0.2       3.75
            ======     ===========     ===   ========  ========

The vdata is composed of 5 fields. 3 records are shown (of course, a vdata
can store much more than that). "INVENTORY" would be the vdata name, and
"partid", "description", etc, would be the field names. The data type varies
between fields. "partid" and "description" would be of "multicharacter" type
(aka "string"), "qty" would be a integer, and "wght" and "price" would be
floats. The text in parentheses could be stored as attributes. A "status"
attribute could be defined for the table as a whole, and given the
value "experimental". Likewise, a "unit" attribute could be associated
with fields "wght" and "price", and given the values "lb" and "$", resp.

The VS API allows one to create, locate and open a vdata inside an
HDF file, update and append records inside it, read records randomly
or sequentially, and access and update the vdata and field attributes.
Attributes can be read and written using the familiar python "dot
notation", and records can be read and written by indexing and slicing the
vdata as if it were a python sequence.

VS module key features
----------------------
VS key features are as follows.

- pyhdf implements almost every routine of the original VS API.
  Only a few have been ignored, most of them being of a rare use:

  - VSgetblocksize() / VSsetblocksize()
  - VSsetnumblocks()
  - VSlone

- It is quite straightforward to go from a C version to a python version
  of a program accessing the VS API, and to learn VS usage by refering to
  the C API documentation.

- A few high-level python methods have been developped to ease
  programmers task. Of greatest interest are the following:

  - Access to attributes through the familiar "dot notation".
  - Indexing and slicing a vdata to read and write its records,
    similarly to a python sequence.
  - Easy retrieval of info on a vdata and its fields.
  - Easy creation of vdatas.

Accessing the VS module
-----------------------
To access the VS module a python program can say one of:

  >>> import pyhdf.VS        # must prefix names with "pyhdf.VS."
  >>> from pyhdf import VS   # must prefix names with "VS."
  >>> from pyhdf.VS import * # names need no prefix

This document assumes the last import style is used.

VS is not self-contained, and needs functionnality provided by another
pyhdf module, namely the HDF module. This module must thus be imported
also:

  >>> from .HDF import *

Package components
------------------
pyhdf is a proper Python package, eg a collection of modules stored under
a directory whose name is that of the package and which stores an
__init__.py file. Following the normal installation procedure, this
directory will be <python-lib>/site-packages/pyhdf', where <python-lib>
stands for the python installation directory.

For each HDF API exists a corresponding set of modules.

The following modules are related to the VS API.

  _hdfext
    C extension module responsible for wrapping the HDF
    C library for all python modules
  hdfext
    python module implementing some utility functions
    complementing the _hdfext extension module
  error
    defines the HDF4Error exception
  HDF
    python module providing support to the VS module
  VS
    python module wrapping the VS API routines inside
    an OOP framework

_hdfext and hdfext were generated using the SWIG preprocessor.
SWIG is however *not* needed to run the package. Those two modules
are meant to do their work in the background, and should never be called
directly. Only HDF and VS should be imported by the user program.

Prerequisites
-------------
The following software must be installed in order for VS to
work.

  HDF (v4) library
    pyhdf does *not* include the HDF4 library, which must
    be installed separately.

    HDF is available at:
    "http://hdf.ncsa.uiuc.edu/obtain.html".

Numeric is also needed by the SD module. See the SD module documentation.

Documentation
-------------
pyhdf has been written so as to stick as closely as possible to
the naming conventions and calling sequences documented inside the
"HDF User s Guide" manual. Even if pyhdf gives an OOP twist
to the C API, the manual can be easily used as a documentary source
for pyhdf, once the class to which a function belongs has been
identified, and of course once requirements imposed by the Python
langage have been taken into account. Consequently, this documentation
will not attempt to provide an exhaustive coverage of the HDF VS
API. For this, the user is referred to the above manual.
The documentation of each pyhdf method will indicate the name
of the equivalent routine as it is found inside the C API.

This document (in both its text and html versions) has been completely
produced using "pydoc", the Python documentation generator (which
made its debut in the 2.1 Python release). pydoc can also be used
as an on-line help tool. For example, to know everything about
the VS.VD class, say:

  >>> from pydoc import help
  >>> from pyhdf.VS import *
  >>> help(VD)

To be more specific and get help only for the read() method of the
VD class:

  >>> help(VD.read)

pydoc can also be called from the command line, as in::

  % pydoc pyhdf.VS.VD         # doc for the whole VD class
  % pydoc pyhdf.VS.VD.read    # doc for the VD.read method

Summary of differences between the pyhdf and C VS API
-----------------------------------------------------
Most of the differences between the pyhdf and C VS API can
be summarized as follows.

- In the C API, every function returns an integer status code, and values
  computed by the function are returned through one or more pointers
  passed as arguments.
- In pyhdf, error statuses are returned through the Python exception
  mechanism, and values are returned as the method result. When the
  C API specifies that multiple values are returned, pyhdf returns a
  sequence of values, which are ordered similarly to the pointers in the
  C function argument list.

Error handling
--------------
All errors reported by the C VS API with a SUCCESS/FAIL error code
are reported by pyhdf using the Python exception mechanism.
When the C library reports a FAIL status, pyhdf raises an HDF4Error
exception (a subclass of Exception) with a descriptive message.
Unfortunately, the C library is rarely informative about the cause of
the error. pyhdf does its best to try to document the error, but most
of the time cannot do more than saying "execution error".

VS needs support from the HDF module
------------------------------------
The VS module is not self-contained (countrary to the SD module).
It requires help from the HDF module, namely:

- the HDF.HDF class to open and close the HDF file, and initialize the
  VS interface
- the HDF.HC class to provide different sorts of constants (opening modes,
  data types, etc).

A program wanting to access HDF vdatas will almost always need to execute
the following minimal set of calls:

  >>> from pyhdf.HDF import *
  >>> from pyhdf.VS import *
  >>> hdfFile = HDF(name, HC.xxx)# open HDF file
  >>> vs = hdfFile.vstart()      # initialize VS interface on HDF file
  >>> ...                        # manipulate vdatas through "vs"
  >>> vs.end()                   # terminate VS interface
  >>> hdfFile.close()            # close HDF file

Classes summary
---------------
pyhdf wraps the VS API using different python classes::

  VS      HDF VS interface
  VD      vdata
  VDField vdata field
  VDattr  attribute (either at the vdata or field level)

In more detail::

  VS     The VS class implements the VS (Vdata) interface applied to an
         HDF file. This class encapsulates the hdf instance, and all
         the top-level functions of the VS API.

         To create a VS instance, call the vstart() method of an
         HDF instance.

         methods:
           constructors:
             attach()       open an existing vdata given its name or
                            reference number, or create a new one,
                            returning a VD instance
             create()       create a new vdata and define its structure,
                            returning a VD instance

           creating and initializing a simple vdata
             storedata()    create a single-field vdata and initialize
                            its values

           closing the interface
             end()          close the VS interface on the HDF file

           searching
             find()         get a vdata reference number given its name
             next()         get the reference number of the vdata following
                            a given one

           inquiry
             vdatainfo()    return info about all the vdatas in the
                            HDF file

  VD     The VD class describes a vdata. It encapsulates
         the VS instance to which the vdata belongs, and the vdata
         identifier.

         To instantiate a VD class, call the attach() or create()
         method of a VS class instance.

         methods:
           constructors
             attr()         create a VDAttr instance representing a
                            vdata attribute; "dot notation" can also be
                            used to access a vdata attribute
             field()        return a VDField instance representing a given
                            field of the vdata

           closing vdata
             detach()       end access to the vdata

           defining fields
             fdefine()      define the name, type and order of a new field
             setfields()    define the field names and field order for
                            the read() and write() methods; also used to
                            initialize the structure of a vdata previously
                            created with the VS.attach() method

           reading and writing
                            note: a vdata can be indexed and sliced like a
                            python sequence

             read()         return the values of a number of records
                            starting at the current record position
             seek()         reset the current record position
             seekend()      seek past the last record
             tell()         return the current record position
             write()        write a number of records starting at the
                            current record position

           inquiry
             attrinfo()     return info about all the vdata attributes
             fexist()       check if a vdata contains a given set of fields
             fieldinfo()    return info about all the vdata fields
             findattr()     locate an attribute, returning a VDAttr instance
                            if found
             inquire()      return info about the vdata
             sizeof()       return the size in bytes of one or more fields

  VDField  The VDField class represents a vdata field. It encapsulates
           the VD instance to which the field belongs, and the field
           index number.

           To instantiate a VDField, call the field() method of a VD class
           instance.

           methods:
             constructors:
               attr()       return a VDAttr instance representing an
                            attribute of the field; "dot notation"
                            can also be used to get/set an attribute.

             inquiry
               attrinfo()   return info about all the field attributes
               find()       locate an attribute, returning a VDAttr
                            instance if found

  VDAttr   The VDAttr class encapsulates methods used to set and query
           attributes defined at the level either of the vdata or the
           vdata field.

           To create an instance of this class, call the attr() or
           findattr() methods of a VD instance (for vdata attributes),
           or call the attr() or find() methods of a VDField instance
           (for field attributes).

           methods:
             get / set
               get()        get the attribute value
               set()        set the attribute value

             info
               info()       retrieve info about the attribute

Data types
----------
Data types come into play when first defining vdata fields and attributes,
and later when querying the definition of those fields and attributes.
Data types are specified using the symbolic constants defined inside the
HC class of the HDF module.

- CHAR and CHAR8 (equivalent): an 8-bit character.
- UCHAR, UCHAR8 and UINT8 (equivalent): unsigned 8-bit values (0 to 255)
- INT8:    signed 8-bit values (-128 to 127)
- INT16:   signed 16-bit values
- UINT16:  unsigned 16 bit values
- INT32:   signed 32 bit values
- UINT32:  unsigned 32 bit values
- FLOAT32: 32 bit floating point values (C floats)
- FLOAT64: 64 bit floating point values (C doubles)

There is no explicit "string" type. To simulate a string, set the field or
attribute type to CHAR, and set the field or attribute "order" to
a value of 'n' > 1. This creates and "array of characters", close
to a string (except that strings will always be of length 'n', right-padded
with spaces if necessary).

Attribute access: low and high level
------------------------------------
The VS API allow setting attributes on vdatas and vdata fields. Attributes
can be of many types (int, float, char) of different bit lengths (8, 16, 32,
64 bits), and can be single or multi-valued. Values of a multi-valued
attribute must all be of the same type.

Attributes can be set and queried in two different ways. First, given a
VD instance (describing a vdata object) or a VDField instance (describing a
vdata field), the attr() method of that instance is called to create a
VDAttr instance representing the wanted attribute (possibly non existent).
The set() method of this VDAttr instance is then called to define the
attribute value, creating it if it does not already exist. The get() method
returns the current attribute value. Here is an example.

  >>> from pyhdf.HDF import *
  >>> from pyhdf.VS import *
  >>> f = HDF('test.hdf', HC.WRITE) # Open file 'test.hdf' in write mode
  >>> vs = f.vstart()            # init vdata interface
  >>> vd = vs.attach('vtest', 1) # attach vdata 'vtest' in write mode
  >>> attr = vd.attr('version')  # prepare to define the 'version' attribute
                                 # on the vdata
  >>> attr.set(HC.CHAR8,'1.0')   # set attribute 'version' to string '1.0'
  >>> print(attr.get())           # get and print attribute value
  >>> fld  = vd.field('fld1')    # obtain a field instance for field 'fld1'
  >>> attr = fld.attr('range')   # prepare to define attribute 'range' on
                                 # this field
  >>> attr.set(HC.INT32,(-10, 15)) # set attribute 'range' to a pair of ints
  >>> print(attr.get())             # get and print attribute value

  >>> vd.detach()                # "close" the vdata
  >>> vs.end()                   # terminate the vdata interface
  >>> f.close()                  # close the HDF file

The second way consists of setting/querying an attribute as if it were a
normal python class attribute, using the usual dot notation. Above example
then becomes:

  >>> from pyhdf.HDF import *
  >>> from pyhdf.VS import *
  >>> f = HDF('test.hdf', HC.WRITE) # Open file 'test.hdf' in write mode
  >>> vs = f.vstart()            # init vdata interface
  >>> vd = vs.attach('vtest', 1) # attach vdata 'vtest' in write mode
  >>> vd.version = '1.0'         # create vdata attribute 'version',
                                 # setting it to string '1.0'
  >>> print(vd.version)           # print attribute value
  >>> fld  = vd.field('fld1')    # obtain a field instance for field 'fld1'
  >>> fld.range = (-10, 15)      # create field attribute 'range', setting
                                 # it to the pair of ints (-10, 15)
  >>> print(fld.range)            # print attribute value
  >>> vd.detach()                # "close" the vdata
  >>> vs.end()                   # terminate the vdata interface
  >>> f.close()                  # close the HDF file

Note how the dot notation greatly simplifies and clarifies the code.
Some latitude is however lost by manipulating attributes in that way,
because the pyhdf package, not the programmer, is then responsible of
setting the attribute type. The attribute type is chosen to be one of:

  =========== ====================================
  HC.CHAR8    if the attribute value is a string
  HC.INT32    if all attribute values are integers
  HC.FLOAT64  otherwise
  =========== ====================================

The first way of handling attribute values must be used if one wants to
define an attribute of any other type (for ex. 8 or 16 bit integers,
signed or unsigned). Also, only a VDAttr instance gives access to attribute
info, through its info() method.

However, accessing HDF attributes as if they were python attributes raises
an important issue. There must exist a way to assign generic attributes
to the python objects without requiring those attributes to be converted
to HDF attributes. pyhdf uses the following rule: an attribute whose name
starts with an underscore ('_') is either a "predefined" attribute
(see below) or a standard python attribute. Otherwise, the attribute
is handled as an HDF attribute. Also, HDF attributes are not stored inside
the object dictionnary: the python dir() function will not list them.

Attribute values can be updated, but it is illegal to try to change the
value type, or the attribute order (number of values). This is important
for attributes holding string values. An attribute initialized with an
'n' character string is simply a character attribute of order 'n' (eg a
character array of length 'n'). If 'vd' is a vdata and we initialize its
'a1' attribute as 'vd.a1 = "abcdef"', then a subsequent update attempt
like 'vd.a1 = "12"' will fail, because we then try to change the order
of the attribute (from 6 to 2). It is mandatory to keep the length of string
attributes constant. Examples below show simple ways how this can be done.

Predefined attributes
---------------------
The VD and VDField classes support predefined attributes to get (and
occasionnaly set) attribute values easily, without having to call a
class method. The names of predefined attributes all start with an
underscore ('_').

In the following tables, the RW column holds an X if the attribute
is read/write. See the HDF User s guide for details about more
"exotic" topics like "class", "faked vdata" and "tag".

  VD predefined attributes

    =========== ==  ========================== =============================
    name        RW  description                C library routine
    =========== ==  ========================== =============================
    _class      X   class name                 VSgetclass/VSsetclass
    _fields         list of field names        VSgetfields
    _interlace  X   interlace mode             VSgetinterlace/VSsetinterlace
    _isattr         true if vdata is "faked"   VSisattr
                    by HDF to hold attributes
    _name       X   name of the vdata          VSgetname/VSsetname
    _nattrs         number of attributes       VSfnattrs
    _nfields        number of fields           VFnfields
    _nrecs          number of records          VSelts
    _recsize        record size (bytes)        VSQueryvsize
    _refnum         reference number           VSQueryref
    _tag            vdata tag                  VSQuerytag
    _tnattrs        total number of vdata and  VSnattrs
                    field attributes
    =========== ==  ========================== =============================

  VDField predefined attributes

    =========== ==  ========================== =============================
    name        RW  description                C library routine
    =========== ==  ========================== =============================
    _esize          external size (bytes)      VFfieldesize
    _index          index number               VSfindex
    _isize          internal size (bytes)      VFfieldisize
    _name           name                       VFfieldname
    _nattrs         number of attributes       VSfnattrs
    _order          order (number of values)   VFfieldorder
    _type           field type (HC.xxx)        VFfieldtype
    =========== ==  ========================== =============================


Record access: low and high level
---------------------------------
vdata records can be read and written in two different ways. The first one
consists of calling the basic I/O methods of the vdata:

- seek() to set the current record position, if necessary;
- read() to retrieve a given number of records from that position;
- write() to write a given number of records starting at
  that position

A second, higher level way, lets one see a vdata similarly to a python
sequence, and access its contents using the familiar indexing and slicing
notation in square brackets. Reading and writing a vdata as if it were a
python sequence may often look simpler, and improve code legibility.

Here are some examples of how a vdata 'vd' holding 3 fields could be read.

  >>> print(vd[0])         # print record 0
  >>> print(vd[-1])        # print last record
  >>> print(vd[2:])        # print records 2 and those that follow
  >>> print(vd[:])         # print all records
  >>> print(vd[:,0])       # print field 0 of all records
  >>> print(vd[:3,:2])     # print first 2 fields of first 3 records

As the above examples show, the usual python rules are obeyed regarding
the interpretation of indexing and slicing values. Note that the vdata
fields can be indexed and sliced, not only the records. The setfields()
method can also be used to select a subset to the vdata fields
(setfields() also let you reorder the fields). When the vdata is
indexed (as opposed to being sliced), a single record is returned as a list
of values. When the vdata is sliced, a list of records is
always returned (thus a 2-level list), even if the slice contains only
one record.

A vdata can also be written similarly to a python sequence. When indexing
the vdata (as opposed to slicing it), a single record must be assigned,
and the record must be given as a sequence of values. It is legal to use
as an index the current number of records in the vdata: the record is then
appended to the vdata. When slicing the vdata, the records assigned to the
slice must always be given as a list of records, even
if only one record is assigned. Also, the number of records assigned must
always match the width of the slice, except if the slice includes or goes
past the last record of the vdata. In that case, the number of records
assigned can exceed the width of the slice, and the extra records are
appended to the vdata. So, to append records to vdata 'vd', simply
assign records to the slice 'vd[vd._nrecs:]'. Note that, even if the
'field' dimension can be specified in the left-hand side expression,
there is no real interest in doing so, since all fields must
be specified when assigning a record to the vdata: it is an error to
try to assign just a few of the fields.

For example, given a vdata 'vd' holding 5 records, and lists 'reca',
'recb', etc, holding record values::

        vd[0] = reca              # updates record 0
        vd[0,:] = reca            # specifying fields is OK, but useless
        vd[0,1:] = reca[1:]       # error: all fields must be assigned
        vd[1] = [recb, recc]      # error: only one record allowed
        vd[5] = recc              # append one record
        vd[1:3] = [reca,recb]     # updates second and third record
        vd[1:4] = [reca, recb]    # error: 3 records needed
        vd[5:] = [reca,recb]      # appends 2 records to the vdata
        vd[4:] = [reca, recb]     # updates last record, append one



Programming models
------------------

Creating and initializing a new vdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following code can serve as a model for the creation and
initialization of a new vdata. It implements the INVENTORY example
described in the "Introduction" section::

    from pyhdf.HDF import *
    from pyhdf.VS import *

    # Open HDF file and initialize the VS interface
    f = HDF('inventory.hdf',    # Open file 'inventory.hdf' in write mode
            HC.WRITE|HC.CREATE) # creating it if it does not exist
    vs = f.vstart()             # init vdata interface

    # Create vdata and define its structure
    vd = vs.create(             # create a new vdata
                   'INVENTORY', # name of the vdata
                                # fields of the vdata follow
               (('partid',HC.CHAR8, 5),       # 5 char string
                ('description',HC.CHAR8, 10), # 10 char string field
                ('qty',HC.INT16, 1),          # 1 16 bit int field
                ('wght',HC.FLOAT32, 1),       # 1 32 bit float
                ('price',HC.FLOAT32,1)        # 1 32 bit float
               ))         # 5 fields allocated in the vdata

    # Set attributes on the vdata and its fields
    vd.field('wght').unit = 'lb'
    vd.field('price').unit = '$'
    # In order to be able to update a string attribute, it must
    # always be set to the same length. This sets 'status' to a 20
    # char long, left-justified string, padded with spaces on the right.
    vd.status = "%-20s" % 'phase 1 done'

    # Store records
    vd.write((                # write 3 records
              ('Q1234', 'bolt',12, 0.01, 0.05),   # record 1
              ('B5432', 'brush', 10, 0.4, 4.25),  # record 2
              ('S7613', 'scissor', 2, 0.2, 3.75)  # record 3
              ))
    vd.detach()               # "close" the vdata

    vs.end()                  # terminate the vdata interface
    f.close()                 # close the HDF file


Note that is mandatory to always write whole records to the vdata.
Note also the comments about the initialization of the 'status'
vdata attribute. We want to be able update this attribute (see
following examples). However, the VS API  prohibits changing an attribute
type when updating its value. Since the length (order) of an attribute
is part of its type, we make sure of setting the attribute to a length
long enough to accomodate the longest possible string we migh want to
assign to the attribute.

Appending records to a vdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Appending records requires first seeking to the end of the vdata, to avoid
overwriting existing records. The following code can serve as a model. The
INVENTORY vdata created before is used::

    from pyhdf.HDF import *
    from pyhdf.VS import *

    f = HDF('inventory.hdf',         # Open 'inventory.hdf' in write mode
            HC.WRITE|HC.CREATE)      # creating it if it does not exist
    vs = f.vstart()                  # init vdata interface
    vd = vs.attach('INVENTORY', 1)   # attach 'INVENTORY' in write mode

    # Update the `status' vdata attribute. The attribute length must not
    # change. We call the attribute info() method, which returns a list
    # where number of values (eg string length) is stored at index 2.
    # We then assign a left justified string of exactly that length.
    len = vd.attr('status').info()[2]
    vd.status = '%-*s' % (len, 'phase 2 done')

    vd[vd._nrecs:] = (                     # append 2 records
          ('A4321', 'axe', 5, 1.5, 25),    # first record
          ('C3214', 'cup', 100, 0.1, 3.25) # second record
                    )
    vd.detach()               # "close" the vdata

    vs.end()                  # terminate the vdata interface
    f.close()                 # close the HDF file

Note how, when updating the value of the 'status' vdata attribute,
we take care of assigning a value of the same length as that of the
original value. Otherwise, the assignment would raise an exception.
Records are written by assigning the vdata through a slicing
expression, like a python sequence. By specifying the number of records
as the start of the slice, the records are appended to the vdata.

Updating records in a vdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Updating requires seeking to the record to update before writing the new
records. New data will overwrite this record and all records that follow,
until a new seek is performed or the vdata is closed. Note that record
numbering starts at 0.

The following code can serve as a model. The INVENTORY vdata created
before is used::

    from pyhdf.HDF import *
    from pyhdf.VS import *

    f = HDF('inventory.hdf',         # Open 'inventory.hdf' in write mode
            HC.WRITE|HC.CREATE)      # creating it if it does not exist
    vs = f.vstart()                  # init vdata interface
    vd = vs.attach('INVENTORY', 1)   # attach 'INVENTORY' in write mode

    # Update the `status' vdata attribute. The attribute length must not
    # change. We call the attribute info() method, which returns a list
    # where number of values (eg string length) is stored at index 2.
    # We then assign a left justified string of exactly that length.
    len = vd.attr('status').info()[2]
    vd.status = '%-*s' % (len, 'phase 3 done')

    # Update record at index 1 (second record)
    vd[1]  = ('Z4367', 'surprise', 10, 3.1, 44.5)
    # Update record at index 4, and all those that follow
    vd[4:] = (
              ('QR231', 'toy', 12, 2.5, 45),
              ('R3389', 'robot', 3, 45, 2000)
              )
    vd.detach()               # "close" the vdata
    vs.end()                  # terminate the vdata interface
    f.close()                 # close the HDF file

Reading a vdata
^^^^^^^^^^^^^^^
The following example shows how read the vdata attributes and sequentially
maneuver through its records. Note how we use the exception mechanism
to break out of the reading loop when we reach the end of the vdata::

    from pyhdf.HDF import *
    from pyhdf.VS import *

    f = HDF('inventory.hdf')         # open 'inventory.hdf' in read mode
    vs = f.vstart()                  # init vdata interface
    vd = vs.attach('INVENTORY')      # attach 'INVENTORY' in read mode

    # Display some vdata attributes
    print "status:", vd.status
    print "vdata: ", vd._name        # predefined attribute: vdata name
    print "nrecs: ", vd._nrecs       # predefined attribute:  num records

    # Display value of attribute 'unit' for all fields on which
    # this attribute is set
    print "units: ",
    for fieldName in vd._fields:     # loop over all field names
        try:
            # instantiate field and obtain value of attribute 'unit'
            v = vd.field(fieldName).unit
            print "%s: %s" % (fieldName, v),
        except:                      # no 'unit' attribute: ignore
            pass
    print ""
    print ""

    # Display table header.
    header = "%-7s %-12s %3s %4s %8s" % tuple(vd._fields)
    print "-" * len(header)
    print header
    print "-" * len(header)

    # Loop over the vdata records, displaying each record as a table row.
    # Current record position is 0 after attaching the vdata.
    while 1:
        try:
            rec = vd.read()       # read next record
            # equivalent to:
          # rec = vd[vd.tell()]
            print "%-7s %-12s %3d %4.1f %8.2f" % tuple(rec[0])
        except HDF4Error:             # end of vdata reached
            break

    vd.detach()               # "close" the vdata
    vs.end()                  # terminate the vdata interface
    f.close()                 # close the HDF file

In the previous example, the reading/displaying loop can be greatly
simplified by rewriting it as follows::

    from pyhdf.HDF import *
    from pyhdf.VS import *

    f = HDF('inventory.hdf')         # open 'inventory.hdf' in read mode
    vs = f.vstart()                  # init vdata interface
    vd = vs.attach('INVENTORY')      # attach 'INVENTORY' in read mode

    ....

    # Read all records at once, and loop over the sequence.
    for rec in vd[:]:
        print "%-7s %-12s %3d %4.1f %8.2f" % tuple(rec)

    vd.detach()               # "close" the vdata
    ...

The indexing expression 'vd[:]' returns the complete set of records,
which can then be looped over using a 'for' statement. This style of loop
is quite clean, and should look very familiar to python adepts.


"""

import os, sys, types

from . import hdfext as _C

from . import six
from .six.moves import xrange
from .HC import HC
from .error import HDF4Error, _checkErr

# List of names we want to be imported by an "from pyhdf.VS import *"
# statement

__all__ = ['VS', 'VD', 'VDField', 'VDAttr']

class VS(object):
    """The VS class implements the VS (Vdata) interface applied to an
    HDF file.
    To instantiate a VS class, call the vstart() method of an
    HDF instance. """

    def __init__(self, hinst):
        # Not to be called directly by the user.
        # A VS object is instantiated using the vstart()
        # method of an HDF instance.

        # Args:
        #    hinst    HDF instance
        # Returns:
        #    A VS instance
        #
        # C library equivalent : Vstart (rather: Vinitialize)

        # Private attributes:
        #  _hdf_inst:       HDF instance

        # Note: Vstart is just a macro; use 'Vinitialize' instead
        status = _C.Vinitialize(hinst._id)
        _checkErr('VS', status, "cannot initialize VS interface")
        self._hdf_inst = hinst


    def __del__(self):
        """Delete the instance, first calling the end() method
        if not already done.          """

        try:
            if self._hdf_inst:
                self.end()
        except:
            pass

    def end(self):
        """Close the VS interface.

        Args::

          No argument

        Returns::

          None

        C library equivalent : Vend
                                                """

        # Note: Vend is just a macro; use 'Vfinish' instead
        _checkErr('end', _C.Vfinish(self._hdf_inst._id),
                  "cannot terminate VS interface")
        self._hdf_inst = None

    vend = end      # For backward compatibility

    def attach(self, num_name, write=0):
        """Locate an existing vdata or create a new vdata in the HDF file,
        returning a VD instance.

        Args::

          num_name  Name or reference number of the vdata. An existing vdata
                    can be specified either through its reference number or
                    its name. Use -1 to create a new vdata.
                    Note that uniqueness is not imposed on vdatas names,
                    whereas refnums are guaranteed to be unique. Thus
                    knowledge of its reference number may be the only way
                    to get at a wanted vdata.

          write     Set to 0 to open the vdata in read-only mode,
                    set to 1 to open it in write mode


        Returns::

          VD instance representing the vdata

        C library equivalent : VSattach

        After creating a new vdata (num_name == -1), fields must be
        defined using method fdefine() of the VD instance, and those
        fields must be allocated to the vdata with method setfields().
        Same results can be achieved, but more simply, by calling the
        create() method of the VS instance.
                                                    """

        mode = write and 'w' or 'r'
        if isinstance(num_name, str):
            num = self.find(num_name)
        else:
            num = num_name
        vd = _C.VSattach(self._hdf_inst._id, num, mode)
        if vd < 0:
            _checkErr('attach', vd, 'cannot attach vdata')
        return VD(self, vd)

    def create(self, name, fields):
        """Create a new vdata, setting its name and allocating
        its fields.

        Args::

          name     Name to assign to the vdata
          fields   Sequence of field definitions. Each field definition
                   is a sequence with the following elements in order:

                   - field name
                   - field type (one of HC.xxx constants)
                   - field order (number of values)

                   Fields are allocated to the vdata in the given order


        Returns::

          VD instance representing the created vdata

        Calling the create() method is equivalent to the following calls:
          - vd = attach(-1,1), to create a new vdata and open it in
                 write mode
          - vd._name = name, to set the vdata name
          - vd.fdefine(...), to define the name, type and order of
                 each field
          - vd.setfields(...), to allocate fields to the vdata

        C library equivalent : no equivalent
                                                      """

        try:
            # Create new vdata (-1), open in write mode (1)
            vd = self.attach(-1, 1)
            # Set vdata name
            vd._name = name
            # Define fields
            allNames = []
            for name, type, order in fields:
                vd.fdefine(name, type, order)
                allNames.append(name)
            # Allocate fields to the vdata
            vd.setfields(*allNames)
            return vd
        except HDF4Error as msg:
            raise HDF4Error("error creating vdata (%s)" % msg)

    def find(self, vName):
        """Get the reference number of a vdata given its name.
        The vdata can then be opened (attached) by passing this
        reference number to the attach() method.

        Args::

          vName    Name of the vdata for which the reference number
                   is needed. vdatas names are not guaranteed to be
                   unique. When more than one vdata bear the same name,
                   find() will return the refnum of the first one founmd.

        Returns::

          vdata reference number. 0 is returned if the vdata does not exist.

        C library equivalent : VSfind
                                               """

        refNum = _C.VSfind(self._hdf_inst._id, vName)
        _checkErr("find", refNum, "cannot find vdata %s" % vName)
        return refNum

    def next(self, vRef):
        """Get the reference number of the vdata following a given
        vdata.

        Args::

          vRef   Reference number of the vdata preceding the one
                 we require. Set to -1 to get the first vdata in
                 the HDF file. Knowing its reference number,
                 the vdata can then be opened (attached) by passing this
                 reference number to the attach() method.

        Returns::

          Reference number of the vdata following the one given
          by argument vref

        An exception is raised if no vdata follows the one given by vRef.

        C library equivalent : VSgetid
                                               """

        num = _C.VSgetid(self._hdf_inst._id, vRef)
        _checkErr('next', num, 'cannot get next vdata')
        return num

    def vdatainfo(self, listAttr=0):
        """Return info about all the file vdatas.

        Args::

          listAttr   Set to 0 to ignore vdatas used to store attribute
                     values, 1 to list them (see the VD._isattr readonly
                     attribute)

        Returns::

          List of vdata descriptions. Each vdata is described as
          a 9-element tuple, composed of the following:

          - vdata name
          - vdata class
          - vdata reference number
          - vdata number of records
          - vdata number of fields
          - vdata number of attributes
          - vdata record size in bytes
          - vdata tag number
          - vdata interlace mode


        C library equivalent : no equivalent
                                                 """

        lst = []
        ref = -1      # start at beginning
        while True:
            try:
                nxtRef = self.next(ref)
            except HDF4Error:    # no vdata left
                break
            # Attach the vdata and check for an "attribute" vdata.
            ref = nxtRef
            vdObj = self.attach(ref)
            if listAttr or not vdObj._isattr:
                # Append a list of vdata properties.
                lst.append((vdObj._name,
                            vdObj._class,
                            vdObj._refnum,
                            vdObj._nrecs,
                            vdObj._nfields,
                            vdObj._nattrs,
                            vdObj._recsize,
                            vdObj._tag,
                            vdObj._interlace))
            vdObj.detach()
        return lst

    def storedata(self, fieldName, values, data_type, vName, vClass):
        """Create and initialize a single field vdata, returning
        the vdata reference number.

        Args::

          fieldName   Name of the single field in the vadata to create
          values      Sequence of values to store in the field;. Each value can
                      itself be a sequence, in which case the field will be
                      multivalued (all second-level sequences must be of
                      the same length)
          data_type   Values type (one of HC.xxx constants). All values
                      must be of the same type
          vName       Name of the vdata to create
          vClass      Vdata class (string)


        Returns::

          vdata reference number

        C library equivalent : VHstoredata / VHstoredatam
                                                """

        # See if the field is multi-valued.
        nrecs = len(values)
        if type(values[0]) in [list, tuple]:
            order = len(values[0])
            # Replace input list with a flattened list.
            newValues = []
            for el in values:
                for e in el:
                    newValues.append(e)
            values = newValues
        else:
            order = 1
        n_values = nrecs * order
        if data_type == HC.CHAR8:
            buf = _C.array_byte(n_values)
            # Allow values to be passed as a string.
            # Noop if a list is passed.
            values = list(values)
            for n in range(n_values):
                values[n] = ord(values[n])

        elif data_type in [HC.UCHAR8, HC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == HC.INT8:
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

        elif data_type == HC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == HC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == HC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == HC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == HC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == HC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("storedata: illegal or unimplemented data_type")

        for n in range(n_values):
            buf[n] = values[n]
        if order == 1:
            vd = _C.VHstoredata(self._hdf_inst._id, fieldName, buf,
                                nrecs, data_type, vName, vClass)
        else:
            vd = _C.VHstoredatam(self._hdf_inst._id, fieldName, buf,
                                nrecs, data_type, vName, vClass, order)

        _checkErr('storedata', vd, 'cannot create vdata')

        return vd


class VD(object):
    """The VD class encapsulates the functionnality of a vdata.
    To instantiate a VD class, call the attach() or the create()
    method of a VS class instance."""

    def __init__(self, vsinst, id):
        # This construtor is not intended to be called directly
        # by the user program. The attach() method of an
        # VS class instance should be called instead.

        # Arg:
        #  vsinst      VS instance from which the call is made
        #  id          vdata reference number

        # Private attributes:
        #  _vs_inst   VS instance to which the vdata belongs
        #  _id        vdata identifier
        #  _offset    current record offset
        #  _setfields last arg to setfields()


        self._vs_inst = vsinst
        self._id = id
        self._offset = 0
        self._setfields = None


    def __getattr__(self, name):
        """Some vdata properties can be queried/set through the following
        attributes. Their names all start with an "_" to avoid
        clashes with user-defined attributes. Most are read-only.
        Only the _class, _fields, _interlace and _name can be modified.
        _fields and _interlace can only be set once.

        Name       RO  Description              C library routine
        -----      --  -----------------        -----------------
        _class         class name               VSgetclass
        _fields    X   field names              VSgetfields
        _interlace     interlace mode           VSgetinterlace
        _isattr    X   attribute vs real vdata  VSisattr
        _name          name                     VSgetname
        _nattrs    X   number of attributes     VSfnattrs
        _nfields   X   number of fields         VFnfields
        _nrecs     X   number of records        VSelts
        _recsize   X   record size              VSQueryvsize
        _refnum    X   reference number         VSQueryref
        _tag       X   tag                      VSQuerytag
        _tnattrs   X   total number of attr.    VSnattrs

                                                         """

        # Check for a user defined attribute first.
        att = self.attr(name)
        if att._index is not None:   # Then the attribute exists
            return att.get()

        # Check for a predefined attribute
        elif name == "_class":
            status, nm = _C.VSgetclass(self._id)
            _checkErr('_class', status, 'cannot get vdata class')
            return nm

        elif name == "_fields":
            n, fields = _C.VSgetfields(self._id)
            _checkErr('_fields', n, "cannot get vdata field names")
            return fields.split(',')

        elif name == "_interlace":
            mode = _C.VSgetinterlace(self._id)
            _checkErr('_interlace', mode, "cannot get vdata interlace mode")
            return mode

        elif name == "_isattr":
            return _C.VSisattr(self._id)

        elif name == "_name":
            status, nm = _C.VSgetname(self._id)
            _checkErr('_name', status, 'cannot get vdata name')
            return nm

        elif name == "_nattrs":
            n = _C.VSfnattrs(self._id, -1)  # -1: vdata attributes
            _checkErr("_nfields", n, "cannot retrieve number of attributes")
            return n

        elif name == "_nfields":
            n = _C.VFnfields(self._id)
            _checkErr("_nfields", n, "cannot retrieve number of fields")
            return n

        elif name == "_nrecs":
            n = _C.VSelts(self._id)
            _checkErr('_nrecs', n, 'cannot get vdata number of records')
            return n

        elif name == "_recsize":
            return self.inquire()[3]

        elif name == "_refnum":
            n = _C.VSQueryref(self._id)
            _checkErr('refnum', n, 'cannot get reference number')
            return n

        elif name == "_tag":
            n = _C.VSQuerytag(self._id)
            _checkErr('_tag', n, 'cannot get tag')
            return n

        elif name == "_tnattrs":
            n = _C.VSnattrs(self._id)
            _checkErr('_tnattrs', n, 'execution error')
            return n

        raise AttributeError

    def __setattr__(self, name, value):

        # A name starting with an underscore will be treated as
        # a standard python attribute, and as an HDF attribute
        # otherwise.

        # Forbid assigning to our predefined attributes
        if name in ["_fields", "_isattr", "_nattrs", "_nfields",
                    "_nrecs", "_recsize", "_refnum", "_tag", "_tnattrs"]:
            raise AttributeError("%s: read-only attribute" % name)

        # Handle the 3 VS attributes: _class, _interlace
        # and _name. _interlace can only be set once.
        elif name == "_class":
            _checkErr(name, _C.VSsetclass(self._id, value),
                      'cannot set _class property')

        elif name == "_interlace":
            _checkErr(name, _C.VSsetinterlace(self._id, value),
                      'cannot set _interlace property')

        elif name == "_name":
            _checkErr(name, _C.VSsetname(self._id, value),
                      'cannot set _name property')

        # Try to set the attribute.
        else:
            _setattr(self, name, value)

    def __getitem__(self, elem):

        # This method is called when the vdata is read
        # like a Python sequence.

        # Parse the indexing expression.
        start, count = self.__buildStartCount(elem)
        # Reset current position if necessary.
        if self._offset != start[0]:
            self.seek(start[0])
        # Get records. A negative count means that an index was used.
        recs = self.read(abs(count[0]))
        # See if all the fields must be returned.
        f0 = start[1]
        if f0 == 0 and count[1] == self._nfields:
            out = recs
        else:
            # Return only a subset of the vdata fields.
            out = []
            f1 = f0 + count[1]
            for r in recs:
                out.append(r[f0:f1])

        # If an index was used (not a slice), return the record as
        # a list, instead of returning it inside a 2-level list,
        if count[0] < 0:
            return out[0]
        return out

    def __setitem__(self, elem, data):

        # This method is called when the vdata is written
        # like a Python sequence.
        #
        # When indexing the vdata, 'data' must specify exactly
        # one record, which must be specifed as a sequence. If the index is
        # equal to the current number of records, the record
        # is appended to the vdata.
        #
        # When slicing the vdata, 'data' must specify a list of records.
        # The number of records in the top level-list must match the width
        # of the slice, except if the slice extends past the end of the
        # vdata. In that case, extra records can be specified in the list,
        # which will be appended to the vdata. In other words,
        # to append records to vdata 'vd', assign records to
        # the slice 'vd[vd._nrecs:]'.
        #
        # For ex., given a vdata 'vd' holding 5 records, and lists
        # 'reca', 'recb', etc holding record values:
        #  vd[0] = reca              # updates record 0
        #  vd[1] = [recb, recc]      # error: only one record allowed
        #  vd[1:3] = [reca,recb]     # updates second and third record
        #  vd[1:4] = [reca, recb]   # error: 3 records needed
        #  vd[5:] = [reca,recb]      # appends 2 records to the vdata

        # Check that arg is a list.
        if not type(data) in [tuple, list]:
            raise HDF4Error("record(s) must be specified as a list")
        start, count = self.__buildStartCount(elem, setitem=1)
        # Records cannot be partially written.
        if start[1] != 0 or count[1] != self._nfields:
            raise HDF4Error("each vdata field must be written")

        # If an index (as opposed to a slice) was applied to the
        # vdata, a single record must be passed. Since write() requires
        # a 2-level list, wrap this record inside a list.
        if count[0] < 0:
            if len(data) != self._nfields:
                raise HDF4Error("record does not specify all fields")
            data = [data]
        # A slice was used. The slice length must match the number of
        # records, except if the end of the slice equals the number
        # of records. Then, extra recors can be specified, which will
        # be appended to the vdata.
        else:
            if count[0] != len(data):
                if start[0] + count[0] != self._nrecs:
                    raise HDF4Error("illegal number of records")
        # Reset current record position if necessary.
        if self._offset != start[0]:
            self.seek(start[0])
        # Write records.
        recs = self.write(data)

    def __del__(self):
        """Delete the instance, first calling the detach() method
        if not already done.          """

        try:
            if self._id:
                self.detach()
        except:
            pass

    def detach(self):
        """Terminate access to the vdata.

        Args::

          no argument

        Returns::

          None

        C library equivalent : VSdetach
                                              """

        _checkErr('detach', _C.VSdetach(self._id), "cannot detach vdata")
        self._id = None

    def fdefine(self, name, type, order):
        """Define a field. To initialize a newly created vdata with
        fields created with fdefine(), assign a tuple of field names
        to the _fields attribute or call the setfields() method.

        Args::

          name     field name
          type     field data type (one of HC.xxx)
          order    field order (number of values in the field)

        Returns::

          None

        C library equivalent : VSfdefine
                                            """

        _checkErr('fdefine', _C.VSfdefine(self._id, name, type, order),
                  'cannot define field')


    def setfields(self, *fldNames):
        """Define the name and order of the fields to access
        with the read() and write() methods.

        Args::

          fldNames  variable length argument specifying one or more
                    vdata field names

        Returns::

          None

        C library equivalent : VSsetfields

        setfields() indicates how to perform the matching between the vdata
        fields and the values passed to the write() method or returned
        by the read() method.

        For example, if the vdata contains fields 'a', 'b' and 'c' and
        a "setfields('c','a')" call is made,  read() will thereafter return
        for each record the values of field 'c' and 'a', in that order.
        Field 'b' will be ignored.

        When writing to a vdata, setfields() has a second usage. It is used
        to initialize the structure of the vdata, that is, the name and order
        of the fields that it will contain. The fields must have been
        previously defined by calls to the fdefine() method.
        Following that first call, setfields() can be called again to
        change the order in which the record values will be passed
        to the write() method. However, since it is mandatory to write
        whole records, subsequent calls to setfields() must specify every
        field name: only the field order can be changed.

                                                   """

        _checkErr('setfields', _C.VSsetfields(self._id, ','.join(fldNames)),
                  'cannot execute')
        self._setfields = fldNames   # remember for read/write routines


    def field(self, name_index):
        """Get a VDField instance representing a field of the vdata.

        Args::

          name_index   name or index number of the field

        Returns::

          VDfield instance representing the field

        C library equivalent : no equivalent
                                                       """

        # Transform a name to an index number
        if isinstance(name_index, str):
            status, index = _C.VSfindex(self._id, name_index)
            _checkErr('field', status, "illegal field name: %s" % name_index)
        else:
            n = _C.VFnfields(self._id)
            _checkErr('field', n, 'cannot execute')
            index = name_index
            if index >= n:
                raise HDF4Error("field: illegal index number")
        return VDField(self, index)


    def seek(self, recIndex):
        """Seek to the beginning of the record identified by its
        record index. A succeeding read will load this record in
        memory.

        Args::

          recIndex  index of the record in the vdata; numbering
                    starts at 0. Legal values range from 0
                    (start of vdata) to the current number of
                    records (at end of vdata).

        Returns::

          record index

        An exception is raised if an attempt is made to seek beyond the
        last record.

        The C API prohibits seeking past the next-to-last record,
        forcing one to read the last record to advance to the end
        of the vdata. The python API removes this limitation.

        Seeking to the end of the vdata can also be done by calling
        method ``seekend()``.

        C library equivalent : VSseek
                                                """

        if recIndex > self._nrecs - 1:
            if recIndex == self._nrecs:
                return self.seekend()
            else:
                raise HDF4Error("attempt to seek past last record")
        n = _C.VSseek(self._id, recIndex)
        _checkErr('seek', n, 'cannot seek')
        self._offset = n
        return n

    def seekend(self):
        """Set the current record position past the last vdata record.
        Subsequent write() calls will append records to the vdata.

        Args::

          no argument

        Returns::

          index of the last record plus 1

        C library equivalent : no equivalent
                                                 """

        try:
            # Seek to the next-to-last record position
            n = self.seek(self._nrecs - 1)       # updates _offset
            # Read last record, ignoring values
            self.read(1)                         # updates _offset
            return self._nrecs
        except HDF4Error:
            raise HDF4Error("seekend: cannot execute")

    def tell(self):
        """Return current record position in the vdata.

        Args::

          no argument

        Returns::

          current record position; 0 is at start of vdata.

        C library equivalent : no equivalent
                                             """

        return self._offset

    def read(self, nRec=1):
        """Retrieve the values of a number of records, starting
        at the current record position. The current record position
        is advanced by the number of records read. Current position
        is 0 after "opening" the vdata with the attach() method.

        Args::

          nRec    number of records to read


        Returns::

          2-level list. First level is a sequence of records,
          second level gives the sequence of values for each record.
          The values returned for each record are those of the fields
          specified in the last call to method setfields(), in that
          order. The complete vdata field set is returned if
          setfields() has not been called.

        An exception is raised if the current record position is
        already at the end of the vdata when read() is called. This
        exception can be caught as an "end of vdata" indication to
        exit a loop which scans each record of the vdata. Otherwise,
        the number of records to be read is lowered to the number of
        records remaining in the vdata, if that number is less than
        the number asked for by parameter 'nRec'. Setting 'nRec' to
        an arbitrarily large value can thus be used to retrieve the
        remaining records in the vdata.

        C library equivalent : VSread
                                                       """
        # Validate number of records to read vs the current offset.
        # Return "end of vdata" exception if already at end of vdata
        # otherwise "clip" the number of records if it exceeds the
        # number of remaining records in the vdata.
        n = self._nrecs
        if self._offset == n:
            raise HDF4Error("end of vdata reached")
        if self._offset + nRec > n:
            nRec = self._offset + nRec - n

        fields = self._setfields or self._fields
        nFields = len(fields)
        fieldList = ','.join(fields)
        _checkErr('read', _C.VSsetfields(self._id, fieldList),
                  'error defining fields to read')

        # Allocate a buffer to store the packed records.
        bufSize = self.sizeof(fields) * nRec
        bigBuf = _C.array_byte(bufSize)

        # Read records
        nRead = _C.VSread(self._id, bigBuf, nRec, 0)   # 0: FULL_INTERLACE
        _checkErr('read', nRead, 'read error')
        self._offset += nRec

        # Allocate an array to store a pointer to the field buffer.
        fldArr = _C.new_array_voidp(1)

        # Initialize return value
        values = []
        for numRec in range(nRead):
            v = []
            for numFld in range(nFields):
                v.append(None)
            values.append(v)

        # Unpack each field in turn.
        for numFld in range(nFields):
            fld = self.field(fields[numFld])
            data_type = fld._type
            order = fld._order
            n_values = order * nRead

            # Allocate a buffer to store the field values.
            if data_type in [HC.CHAR8, HC.UCHAR8, HC.UINT8]:
                buf = _C.array_byte(n_values)

            elif data_type == HC.INT8:
                buf = _C.array_int8(n_values)

            elif data_type == HC.INT16:
                buf = _C.array_int16(n_values)

            elif data_type == HC.UINT16:
                buf = _C.array_uint16(n_values)

            elif data_type == HC.INT32:
                buf = _C.array_int32(n_values)

            elif data_type == HC.UINT32:
                buf = _C.array_uint32(n_values)

            elif data_type == HC.FLOAT32:
                buf = _C.array_float32(n_values)

            elif data_type == HC.FLOAT64:
                buf = _C.array_float64(n_values)

            else:
                raise HDF4Error("read: illegal or unupported type %d" % \
                                 data_type)

            # Unpack the field values.
            _C.array_voidp_setitem(fldArr, 0, buf)
            _checkErr('read',
                      _C.VSfpack(self._id, 1, fieldList, bigBuf, bufSize,
                                 nRead, fld._name, fldArr),
                      "cannot execute")

            # Extract values from the field buffer.
            k = 0
            for numRec in range(nRead):
                if order == 1:
                    values[numRec][numFld] = buf[k]
                    k += 1
                else:
                    # Handle strings specially
                    if data_type == HC.CHAR8:
                        s = ''
                        for i in range(order):
                            v = buf[k]
                            if v != 0:
                                s += chr(v)
                            k += 1
                        values[numRec][numFld] = s
                    # Return field values as a list
                    else:
                        values[numRec][numFld] = []
                        for i in range(order):
                            values[numRec][numFld].append(buf[k])
                            k += 1

            del buf

        return values


    def write(self, values):
        """Write records to the vdata. Writing starts at the current
        record position, which is advanced by the number of records
        written.

        Args::

          values: 2-level sequence. First level is a sequence of records.
                  A second level gives the sequence of record values.
                  It is mandatory to always write whole records. Thus
                  every record field must appear at the second level.
                  The record values are ordered according the list of
                  field names set in the last call to the setfields()
                  method. The ordre of the complete vdata field set is
                  used if setfields() has not been called.


        Returns::

          number of records written

        To append to a vdata already holding 'n' records, it is necessary
        to first move the current record position to 'n-1' with a call to
        method seek(), then to call method read() for the side effect
        of advancing the current record position past this last record.
        Method seekend() does just that.

        C library equivalent : VSwrite
                                                       """

        nFields = self._nfields
        # Fields give the order the record values, as defined in the
        # last call to setfields()
        fields = self._setfields or self._fields
        # We must pack values using the effective field order in the vdata
        fieldList = ','.join(self._fields)

        # Validate the values argument.
        if nFields != len(fields):
            raise HDF4Error("write: must write whole records")
        if type(values) not in [list, tuple]:
            raise HDF4Error("write: values must be a sequence")
        nRec = len(values)
        for n in range(nRec):
            rec = values[n]
            if type(rec) not in [list, tuple]:
                raise HDF4Error("write: records must be given as sequences")
            # Make sure each record is complete.
            if len(rec) != nFields:
                raise HDF4Error("write: records must specify every field")

        # Allocate a buffer to store the packed records.
        bufSize = self._recsize * nRec
        bigBuf = _C.array_byte(bufSize)

        # Allocate an array to store a pointer to the field buffer.
        fldArr = _C.new_array_voidp(1)

        # Pack each field in turn.
        for numFld in range(nFields):
            fld = self.field(fields[numFld])
            data_type = fld._type
            order = fld._order
            n_values = order * nRec

            # Allocate a buffer to store the field values.
            if data_type in [HC.CHAR8, HC.UCHAR8, HC.UINT8]:
                buf = _C.array_byte(n_values)

            elif data_type == HC.INT8:
                buf = _C.array_int8(n_values)

            elif data_type == HC.INT16:
                buf = _C.array_int16(n_values)

            elif data_type == HC.UINT16:
                buf = _C.array_uint16(n_values)

            elif data_type == HC.INT32:
                buf = _C.array_int32(n_values)

            elif data_type == HC.UINT32:
                buf = _C.array_uint32(n_values)

            elif data_type == HC.FLOAT32:
                buf = _C.array_float32(n_values)

            elif data_type == HC.FLOAT64:
                buf = _C.array_float64(n_values)

            else:
                raise HDF4Error("write: illegal or unupported type %d" % \
                                 data_type)

            # Load the field buffer with values.
            k = 0
            for numRec in range(nRec):
                val = values[numRec][numFld]
                # Single-valued field
                if order == 1:
                    buf[k] = val
                    k += 1
                # Multi-valued field
                else:
                    # Handle strings specially.
                    if data_type == HC.CHAR8:
                        if not isinstance(val, str):
                            raise HDF4Error("char fields must be set with strings")
                        n = len(val)
                        for i in range(order):
                            buf[k] = i < n and ord(val[i]) or 0
                            k += 1
                    # Should validate field elements ...
                    elif type(val) not in [list, tuple]:
                        raise HDF4Error("multi-values fields must be given as sequences")
                    else:
                        for i in range(order):
                            buf[k] = val[i]
                            k += 1

            # Store address of the field buffer in first position
            # of the field array. Pack the field values.
            _C.array_voidp_setitem(fldArr, 0, buf) # fldArr[0] = buf
            _checkErr('write',
                      _C.VSfpack(self._id, 0, fieldList, bigBuf, bufSize,
                                 nRec, fld._name, fldArr),
                      "cannot execute")
            del buf

        # Write the packed records.
        n = _C.VSwrite(self._id, bigBuf, nRec, 0)   # 0: FULL_INTERLACE
        _checkErr('write', n, 'cannot execute')
        self._offset += nRec

        return n

    def inquire(self):
        """Retrieve info about the vdata.

        Args::

          no argument

        Returns::

          5-element tuple with the following elements:
            -number of records in the vdata
            -interlace mode
            -list of vdata field names
            -size in bytes of the vdata record
            -name of the vdata

        C library equivalent : VSinquire
                                             """

        status, nRecs, interlace, fldNames, size, vName = \
                _C.VSinquire(self._id)
        _checkErr('inquire', status, "cannot query vdata info")
        return nRecs, interlace, fldNames.split(','), size, vName


    def fieldinfo(self):
        """Retrieve info about all vdata fields.

        Args::

          no argument

        Returns::

          list where each element describes a field of the vdata;
          each field is described by an 7-element tuple containing
          the following elements:

          - field name
          - field data type (one of HC.xxx constants)
          - field order
          - number of attributes attached to the field
          - field index number
          - field external size
          - field internal size

        C library equivalent : no equivalent
                                                      """

        lst = []
        for n in range(self._nfields):
            fld = self.field(n)
            lst.append((fld._name,
                        fld._type,
                        fld._order,
                        fld._nattrs,
                        fld._index,
                        fld._esize,
                        fld._isize))

        return lst

    def sizeof(self, fields):
        """Retrieve the size in bytes of the given fields.

        Args::

          fields   sequence of field names to query

        Returns::

          total size of the fields in bytes

        C library equivalent : VSsizeof
                                                   """

        if type(fields) in [tuple, list]:
            str = ','.join(fields)
        else:
            str = fields
        n = _C.VSsizeof(self._id, str)
        _checkErr('sizeof', n, "cannot retrieve field sizes")
        return n

    def fexist(self, fields):
        """Check if a vdata contains a given set of fields.

        Args::

          fields   sequence of field names whose presence in the
                   vdata must be checked

        Returns::

          true  (1) if the given fields are present
          false (0) otherwise

        C library equivalent : VSfexist
                                                         """

        if type(fields) in [tuple, list]:
            str = ','.join(fields)
        else:
            str = fields
        ret = _C.VSfexist(self._id, str)
        if ret < 0:
            return 0
        else:
            return 1

    def attr(self, name_or_index):
        """Create a VDAttr instance representing a vdata attribute.

        Args::

          name_or_index   attribute name or index number; if a name is
                          given, the attribute may not exist; in that
                          case, it will be created when the VSAttr
                          instance set() method is called

        Returns::

          VSAttr instance for the attribute. Call the methods of this
          class to query, read or set the attribute.

        C library equivalent : no equivalent

                                """

        return VDAttr(self, name_or_index, -1)   # -1: vdata attribute

    def findattr(self, name):
        """Search the vdata for a given attribute.

        Args::

          name    attribute name

        Returns::

          if found, VDAttr instance describing the attribute
          None otherwise

         C library equivalent : VSfindattr
                                                  """

        try:
            att = self.attr(name)
            if att._index is None:
                att = None
        except HDF4Error:
            att = None
        return att

    def attrinfo(self):
        """Return info about all the vdata attributes.

        Args::

          no argument

        Returns::

          dictionnary describing each vdata attribute; for each attribute
          a (name,data) pair is added to the dictionary, where 'data' is
          a tuple holding:
          - attribute data type (one of HC.xxx constants)
          - attribute order
          - attribute value
          - attribute size in bytes

        C library equivalent : no equivalent
                                                  """

        dic = {}
        for n in range(self._nattrs):
            att = self.attr(n)
            name, type, order, size = att.info()
            dic[name] = (type, order, att.get(), size)

        return dic

    def __buildStartCount(self, elem, setitem=0):

        # Called by __getitem__() and __setitem__() methods
        # to parse the expression used inside square brackets to
        # index/slice a vdata.
        # If 'setitem' is set, the call comes from __setitem__()
        # We then allow the start value to be past the last record
        # so as to be able to append to the vdata.
        #
        # Return a 2-element tuple:
        #  - tuple of the start indices along the vdata dimensions
        #  - tuple of the count values along the vdata dimensions
        #    a count of -1 indicates that an index, not a slice
        #    was applied on the correcponding dimension.

        # Make sure the indexing expression does not exceed the
        # vdata number of dimensions (2).
        if isinstance(elem, tuple):
            if len(elem) > 2:
                raise HDF4Error("illegal indexing expression")
        else:    # Convert single index to sequence
            elem = [elem]

        start = []
        count = []
        shape = [self._nrecs, self._nfields]
        n = -1
        for e in elem:
            n += 1
            # Simple index
            if isinstance(e, int):
                is_slice = False
                if e < 0:
                    e += shape[n]
                if e < 0 or e >= shape[n]:
                    if e == shape[n] and setitem:
                        pass
                    else:
                        raise HDF4Error("index out of range")
                beg = e
                end = e + 1
            # Slice index
            elif isinstance(e, slice):
                is_slice = True
                # None or 0 means not specified
                if e.start:
                    beg = e.start
                    if beg < 0:
                        beg += shape[n]
                else:
                    beg = 0
                # None or maxint means not specified
                if e.stop and e.stop != sys.maxsize:
                    end = e.stop
                    if end < 0:
                        end += shape[n]
                else:
                    end = shape[n]
            # Bug
            else:
                raise ValueError("invalid indexing expression")

            # Clip end index and compute number of elements to get
            if end > shape[n]:
                end = shape[n]
            if beg > end:
                beg = end
            if is_slice:
                cnt = end - beg
            else:
                cnt = -1
            start.append(beg)
            count.append(cnt)
        if n == 0:
            start.append(0)
            count.append(shape[1])

        return start, count

class VDField(object):
    """The VDField class represents a vdata field.
    To create a VDField instance, call the field() method of a
    VD class instance. """

    def __init__(self, vdinst, fIndex):
        # This method should not be called directly by the user program.
        # To create a VDField instance, obtain a VD class instance and
        # call its field() method.

        # Args:
        #  vdinst    VD instance to which the field belongs
        #  fIndex    field index
        #
        # Private attributes:
        #  _vd_inst  VD instance to which the field belongs
        #  _idx      field index

        self._vd_inst = vdinst
        self._idx = fIndex


    def __getattr__(self, name):
        """Some field properties can be queried through the following
        read-only attributes. Their names all start with an "_" to avoid
        clashes with user-defined attributes.

        Name      Description              C library routine
        -----     -------------------      -----------------
        _esize     field external size      VFfieldesize
        _index     field index number       VSfindex
        _isize     field internal size      VFfieldisize
        _name      field name               VFfieldname
        _nattrs    number of attributes     VSfnattrs
        _order     field order              VFfieldorder
        _type      field type               VFfieldtype

                                                                   """
        # Check for a user defined attribute first.
        att = self.attr(name)
        if att._index is not None:   # Then the attribute exists
            return att.get()

        # Check for a predefined attribute.
        elif name == "_esize":
            n = _C.VFfieldesize(self._vd_inst._id, self._idx)
            _checkErr('_esize', n, "execution error")
            return n

        elif name == "_index":
            return self._idx

        elif name == "_isize":
            n = _C.VFfieldisize(self._vd_inst._id, self._idx)
            _checkErr('_isize', n, "execution error")
            return n

        elif name == "_name":
            n = _C.VFfieldname(self._vd_inst._id, self._idx)
            _checkErr('_name', n, "execution error")
            return n

        elif name == "_nattrs":
            n = _C.VSfnattrs(self._vd_inst._id, self._idx)
            _checkErr('_nattrs', n, "execution error")
            return n

        elif name == "_order":
            n = _C.VFfieldorder(self._vd_inst._id, self._idx)
            _checkErr('_order', n, "execution error")
            return n

        elif name == "_type":
            type = _C.VFfieldtype(self._vd_inst._id, self._idx)
            _checkErr('_type', type, 'cannot retrieve field type')
            return type

        raise AttributeError


    def __setattr__(self, name, value):

        # Forbid assigning to our predefined attributes
        if name in ["_esize", "_index", "_isize", "_name",
                    "_nattrs", "_order", "_type"]:
            raise AttributeError("%s: read-only attribute" % name)

        # Try to set the attribute.
        else:
            _setattr(self, name, value)

    def attr(self, name_or_index):
        """Create a VDAttr instance representing a field attribute.

        Args::

          name_or_index   attribute name or index number; if a name is
                          specified, the attribute may not exist; in that
                          case, it will be created when the VDAttr
                          instance set() method is called; if an
                          index number is specified, the attribute
                          must exist

        Returns::

          VSAttr instance for the attribute. Call the methods of this
          class to query, read or set the attribute.

        C library equivalent : no equivalent

                                """

        return VDAttr(self, name_or_index, self._idx)

    def find(self, name):
        """Search the field for a given attribute.

        Args::

          name    attribute name

        Returns::

          if found, VDAttr instance describing the attribute
          None otherwise

         C library equivalent : VSfindattr
                                                  """

        try:
            att = self.attr(name)
            if att._index is None:
                att = None
        except HDF4Error:
            att = None
        return att

    def attrinfo(self):
        """Return info about all the field attributes.

        Args::

          no argument

        Returns::

          dictionnary describing each vdata attribute; for each attribute
          a (name,data) pair is added to the dictionary, where 'data' is
          a tuple holding:

          - attribute data type (one of HC.xxx constants)
          - attribute order
          - attribute value
          - attribute size in bytes

        C library equivalent : no equivalent
                                                  """

        dic = {}
        for n in range(self._nattrs):
            att = self.attr(n)
            name, type, order, size = att.info()
            dic[name] = (type, order, att.get(), size)

        return dic


class VDAttr(object):
    """The VDAttr class encapsulates methods used to set and query attributes
    defined at the level either of the vdata or of the vdata field.
    To create an instance of this class, call the attr() method of a VD
    (vdata) or VDField (vdata field) instance. """

    def __init__(self, obj, name_or_index, fIndex):
        # This constructor should not be called directly by the user
        # program. The attr() method of a VD (vdata) or VDField
        # (vdata field) must be called to instantiate this class.

        # Args:
        #  obj            object instance (VD or VDField) to which the
        #                 attribute belongs
        #  name_or_index  name or index of the attribute; if a name is
        #                 given, an attribute with that name will be
        #                 searched, if not found, a new index number will
        #                 be generated
        #  fIndex         field index, or -1 if the attribute belongs
        #                 to the vdata

        # Private attributes:
        #  _vd_inst       VD instance
        #  _vdf_inst      VDField instance or None
        #  _index         attribute index or None
        #  _name          attribute name or None
        #  _fIndex        field index, or -1 obj is a VD instance

        if isinstance(obj, VD):
            self._vd_inst = obj
            self._vdf_instance = None
            self._fIndex = -1
        else:
            self._vd_inst = obj._vd_inst
            self._vdf_inst = obj
            self._fIndex = fIndex
        # Name is given. Attribute may exist or not.
        if isinstance(name_or_index, type('')):
            self._name = name_or_index
            self._index = _C.VSfindattr(self._vd_inst._id, self._fIndex,
                                        self._name);
            if self._index < 0:
                self._index = None
        # Index is given. Attribute Must exist.
        else:
            self._index = name_or_index
            status, self._name, data_type, n_values, size = \
                    _C.VSattrinfo(self._vd_inst._id, self._fIndex,
                                  self._index)
            _checkErr('attr', status, 'non-existent attribute')

    def get(self):
        """Retrieve the attribute value.

        Args::

          no argument

        Returns::

          attribute value(s); a list is returned if the attribute
          is made up of more than one value, except in the case of a
          string-valued attribute (data type HC.CHAR8) where the
          values are returned as a string

        C library equivalent : VSgetattr

                                                """
        # Make sure th attribute exists.
        if self._index is None:
            raise HDF4Error("non existent attribute")
        # Obtain attribute type and the number of values.
        status, aName, data_type, n_values, size = \
                    _C.VSattrinfo(self._vd_inst._id, self._fIndex,
                                  self._index)
        _checkErr('get', status, 'illegal parameters')

        # Get attribute value.
        convert = _array_to_ret
        if data_type == HC.CHAR8:
            buf = _C.array_byte(n_values)
            convert = _array_to_str

        elif data_type in [HC.UCHAR8, HC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == HC.INT8:
            buf = _C.array_int8(n_values)

        elif data_type == HC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == HC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == HC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == HC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == HC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == HC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("get: attribute index %d has an "\
                             "illegal or unupported type %d" % \
                             (self._index, data_type))

        status = _C.VSgetattr(self._vd_inst._id, self._fIndex,
                              self._index, buf)
        _checkErr('get', status, 'illegal attribute ')
        return convert(buf, n_values)

    def set(self, data_type, values):
        """Set the attribute value.

        Args::

          data_type    : attribute data type (see constants HC.xxx)
          values       : attribute value(s); specify a list to create
                         a multi-valued attribute; a string valued
                         attribute can be created by setting 'data_type'
                         to HC.CHAR8 and 'values' to the corresponding
                         string

                         If the attribute already exists, it will be
                         updated. However, it is illegal to try to change
                         its data type or its order (number of values).

        Returns::

          None

        C library equivalent : VSsetattr

                                                  """
        try:
            n_values = len(values)
        except:
            values = [values]
            n_values = 1
        if data_type == HC.CHAR8:
            buf = _C.array_byte(n_values)
            # Allow values to be passed as a string.
            # Noop if a list is passed.
            values = list(values)
            for n in range(n_values):
                if not isinstance(values[n], int):
                    values[n] = ord(values[n])

        elif data_type in [HC.UCHAR8, HC.UINT8]:
            buf = _C.array_byte(n_values)

        elif data_type == HC.INT8:
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

        elif data_type == HC.INT16:
            buf = _C.array_int16(n_values)

        elif data_type == HC.UINT16:
            buf = _C.array_uint16(n_values)

        elif data_type == HC.INT32:
            buf = _C.array_int32(n_values)

        elif data_type == HC.UINT32:
            buf = _C.array_uint32(n_values)

        elif data_type == HC.FLOAT32:
            buf = _C.array_float32(n_values)

        elif data_type == HC.FLOAT64:
            buf = _C.array_float64(n_values)

        else:
            raise HDF4Error("set: illegal or unimplemented data_type")

        for n in range(n_values):
            buf[n] = values[n]
        status = _C.VSsetattr(self._vd_inst._id, self._fIndex, self._name,
                              data_type, n_values, buf)
        _checkErr('attr', status, 'cannot execute')
        # Update the attribute index
        self._index = _C.VSfindattr(self._vd_inst._id, self._fIndex,
                                    self._name);
        if self._index < 0:
            raise HDF4Error("set: error retrieving attribute index")

    def info(self):
        """Retrieve info about the attribute.

        Args::

          no argument

        Returns::

          4-element tuple with the following components:
            -attribute name
            -attribute data type (one of HC.xxx constants)
            -attribute order (number of values)
            -attribute size in bytes

        C library equivalent : VSattrinfo
                                                           """

        # Make sure the attribute exists.
        if self._index is None:
            raise HDF4Error("non existent attribute")

        status, name, type, order, size = \
                _C.VSattrinfo(self._vd_inst._id, self._fIndex, self._index)
        _checkErr('info', status, "execution error")
        return name, type, order, size


###########################
# Support functions
###########################


def _setattr(obj, name, value):
    # Called by the __setattr__ method of the VD and VDField objects.
    #
    #  obj   instance on which the attribute is set
    #  name  attribute name
    #  value attribute value

    if isinstance(value, six.string_types):
        value = value.encode('utf8')

    # Treat a name starting with and underscore as that of a
    # standard python instance attribute.
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
               not bytes in typeList:
            if t not in typeList:
                typeList.append(t)
        # Prohibit sequence of strings or a mix of numbers and string.
        elif t == bytes and not typeList:
            typeList.append(t)
        else:
            typeList = []
            break
    if bytes in typeList:
        xtype = HC.CHAR8
        value = value[0]
    # double is "stronger" than int
    elif float in typeList:
        xtype = HC.FLOAT64
    elif int in typeList:
        xtype = HC.INT32
    else:
        raise HDF4Error("Illegal attribute value")

    # Assign value
    try:
        a = obj.attr(name)
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
    # Strip NULL at end
    if chrs[-1] == '\0':
        del chrs[-1]
    return ''.join(chrs)
