# $Id: V.py,v 1.2 2005-07-14 01:36:41 gosselin_a Exp $
# $Log: not supported by cvs2svn $
# Revision 1.1  2004/08/02 15:36:04  gosselin
# Initial revision
#
# Author: Andre Gosselin
#         Maurice-Lamontagne Institute
#         gosselina@dfo-mpo.gc.ca

"""
V (Vgroup) API (:mod:`pyhdf.V`)
===============================

A module of the pyhdf package implementing the V (Vgroup)
API of the NCSA HDF4 library.
(see: hdf.ncsa.uiuc.edu)

Introduction
------------
V is one of the modules composing pyhdf, a python package implementing
the NCSA HDF library and letting one manage HDF files from within a python
program. Two versions of the HDF library currently exist, version 4 and
version 5. pyhdf only implements version 4 of the library. Many
different APIs are to be found inside the HDF4 specification.
Currently, pyhdf implements just a few of those: the SD, VS and V APIs.
Other APIs should be added in the future (GR, AN, etc).

The V API supports the definition of vgroups inside an HDF file. A vgroup
can thought of as a collection of arbitrary "references" to other HDF
objects defined in the same file. A vgroup may hold references to
other vgroups. It is thus possible to organize HDF objects into some sort
of a hierarchy, similar to files grouped into a directory tree under unix.
This vgroup hierarchical nature partly explains the origin of the "HDF"
name (Hierarchical Data Format). vgroups can help logically organize the
contents of an HDF file, for example by grouping together all the datasets
belonging to a given experiment, and subdividing those datasets according
to the day of the experiment, etc.

The V API provides functions to find and access an existing vgroup,
create a new one, delete a vgroup, identify the members of a vgroup, add
and remove members to and from a vgroup, and set and query attributes
on a vgroup. The members of a vgroup are identified through their tags
and reference numbers. Tags are constants identifying each main object type
(dataset, vdata, vgroup). Reference numbers serve to distinguish among
objects of the same type. To add an object to a vgroup, one must first
initialize that object using the API proper to that object type (eg: SD for
a dataset) so as to create a reference number for that object, and then
pass this reference number and the type tag to the V API. When reading the
contents of a vgroup, the V API returns the tags and reference numbers of
the objects composing the vgroup. The user program must then call the
proper API to process each object, based on tag of this object (eg: VS for
a tag identifying a vdata object).

Some limitations of the V API must be stressed. First, HDF imposes
no integrity constraint whatsoever on the contents of a vgroup, nor does it
help maintain such integrity. For example, a vgroup is not strictly
hierarchical, because an object can belong to more than one vgroup. It would
be easy to create vgroups showing cycles among their members. Also, a vgroup
member is simply a reference to an HDF object. If this object is afterwards
deleted for any reason, the vgroup membership will not be automatically
updated. The vgroup will refer to a non-existent object and thus be left
in an inconsistent state. Nothing prevents adding the same member more than
once to a vgroup, and giving the same name to more than one vgroup.
Finally, the HDF library seems to make heavy use of vgroups for its own
internal needs, and creates vgroups "behind the scenes". This may make it
difficult to pick up "user defined" vgroups when browsing an HDF file.

Accessing the V module
-----------------------
To access the V module a python program can say one of:

  >>> import pyhdf.V           # must prefix names with "pyhdf.V."
  >>> from pyhdf import V      # must prefix names with "V."
  >>> from pyhdf.V import *    # names need no prefix

This document assumes the last import style is used.

V is not self-contained, and needs functionnality provided by another
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

The following modules are related to the V API.

  _hdfext
    C extension module responsible for wrapping the HDF
    C library for all python modules
  hdfext
    python module implementing some utility functions
    complementing the _hdfext extension module
  error
    defines the HDF4Error exception
  HDF
    python module providing support to the V module
  V
    python module wrapping the V API routines inside
    an OOP framework

_hdfext and hdfext were generated using the SWIG preprocessor.
SWIG is however *not* needed to run the package. Those two modules
are meant to do their work in the background, and should never be called
directly. Only HDF and V should be imported by the user program.


Prerequisites
-------------
The following software must be installed in order for the V module to
work.

  HDF (v4) library
    pyhdf does *not* include the HDF4 library, which must
    be installed separately.

    HDF is available at:
    "http://hdf.ncsa.uiuc.edu/obtain.html".

Numeric is also needed by the SD module. See the SD module documentation.


Summary of differences between the pyhdf and C V API
-----------------------------------------------------
Most of the differences between the pyhdf and C V API can
be summarized as follows.

   -In the C API, every function returns an integer status code, and values
    computed by the function are returned through one or more pointers
    passed as arguments.
   -In pyhdf, error statuses are returned through the Python exception
    mechanism, and values are returned as the method result. When the
    C API specifies that multiple values are returned, pyhdf returns a
    sequence of values, which are ordered similarly to the pointers in the
    C function argument list.

Error handling
--------------
All errors reported by the C V API with a SUCCESS/FAIL error code
are reported by pyhdf using the Python exception mechanism.
When the C library reports a FAIL status, pyhdf raises an HDF4Error
exception (a subclass of Exception) with a descriptive message.
Unfortunately, the C library is rarely informative about the cause of
the error. pyhdf does its best to try to document the error, but most
of the time cannot do more than saying "execution error".

V needs support from the HDF module
------------------------------------
The VS module is not self-contained (countrary to the SD module).
It requires help from the HDF module, namely:

- the HDF.HDF class to open and close the HDF file, and initialize the
  V interface
- the HDF.HC class to provide different sorts of constants (opening modes,
  data types, etc).

A program wanting to access HDF vgroups will almost always need to execute
the following minimal set of calls:

  >>> from pyhdf.HDF import *
  >>> from pyhdf.V import *
  >>> hdfFile = HDF(name, HC.xxx)# open HDF file
  >>> v = hdfFile.vgstart()      # initialize V interface on HDF file
  >>> ...                        # manipulate vgroups
  >>> v.end()                    # terminate V interface
  >>> hdfFile.close()            # close HDF file


Classes summary
---------------

pyhdf wraps the V API using the following python classes::

  V      HDF V interface
  VG     vgroup
  VGAttr vgroup attribute

In more detail::

  V     The V class implements the V (Vgroup) interface applied to an
        HDF file.

        To instantiate a V class, call the vgstart() method of an
        HDF instance.

        methods:
          constructors
            attach()      open an existing vgroup given its name or its
                          reference number, or create a new vgroup,
                          returning a VG instance for that vgroup
            create()      create a new vgroup, returning a VG instance
                          for that vgroup

          closing the interface
            end()         close the V interface on the HDF file

          deleting a vgroup
            delete()      delete the vgroup identified by its name or
                          its reference number

          searching
            find()        find a vgroup given its name, returning
                          the vgroup reference number
            findclass()   find a vgroup given its class name, returning
                          the vgroup reference number
            getid()       return the reference number of the vgroup
                          following the one with the given reference number

  VG    The VG class encapsulates the functionnality of a vgroup.

        To instantiate a VG class, call the attach() or create() methods
        of a V class instance.

          constructors

            attr()        return a VGAttr instance representing an attribute
                          of the vgroup
            findattr()    search the vgroup for a given attribute,
                          returning a VGAttr instance for that attribute

          ending access to a vgroup

            detach()      terminate access to the vgroup

          adding a member to a vgroup

            add()         add to the vgroup the HDF object identified by its
                          tag and reference number
            insert()      insert a vdata or a vgroup in the vgroup, given
                          the vdata or vgroup instance

          deleting a member from a vgroup

            delete()      remove from the vgroup the HDF object identified
                          by the given tag and reference number

          querying vgroup

            attrinfo()    return info about all the vgroup attributes
            inqtagref()   determine if the HDF object with the given
                          tag and reference number belongs to the vgroup
            isvg()        determine if the member with the given reference
                          number is a vgroup object
            isvs()        determine if the member with the given reference
                          number is a vdata object
            nrefs()       return the number of vgroup members with the
                          given tag
            tagref()      get the tag and reference number of a vgroup
                          member, given the index number of that member
            tagrefs()     get the tags and reference numbers of all the
                          vgroup members

  VGAttr The VGAttr class provides methods to set and query vgroup
         attributes.

         To create an instance of this class, call the attr() method
         of a VG instance.

         Remember that vgroup attributes can also be set and queried by
         applying the standard python "dot notation" on a VG instance.

           get attibute value(s)

             get()         obtain the attribute value(s)

           set attribute value(s)

             set()         set the attribute to the given value(s) of the
                           given type, first creating the attribute if
                           necessary

           query attribute info

             info()        retrieve attribute name, data type, order and
                           size

Attribute access: low and high level
------------------------------------
The V API allows setting attributes on vgroups. Attributes can be of many
types (int, float, char) of different bit lengths (8, 16, 32, 64 bits),
and can be single or multi-valued. Values of a multi-valued attribute must
all be of the same type.

Attributes can be set and queried in two different ways. First, given a
VG instance (describing a vgroup object), the attr() method of that instance
is called to create a VGAttr instance representing the wanted attribute
(possibly non existent). The set() method of this VGAttr instance is then
called to define the attribute value, creating it if it does not already
exist. The get() method returns the current attribute value. Here is an
example.

  >>> from pyhdf.HDF import *
  >>> from pyhdf.V import *
  >>> f = HDF('test.hdf', HC.WRITE) # Open file 'test.hdf' in write mode
  >>> v = f.vgstart()            # init vgroup interface
  >>> vg = v.attach('vtest', 1)  # attach vgroup 'vtest' in write mode
  >>> attr = vg.attr('version')  # prepare to define the 'version' attribute
                                 # on the vdata
  >>> attr.set(HC.CHAR8,'1.0')   # set attribute 'version' to string '1.0'
  >>> print(attr.get())           # get and print attribute value
  >>> attr = vg .attr('range')   # prepare to define attribute 'range'
  >>> attr.set(HC.INT32,(-10, 15)) # set attribute 'range' to a pair of ints
  >>> print(attr.get())             # get and print attribute value

  >>> vg.detach()                # "close" the vgroup
  >>> v.end()                    # terminate the vgroup interface
  >>> f.close()                  # close the HDF file

The second way consists of setting/querying an attribute as if it were a
normal python class attribute, using the usual dot notation. Above example
then becomes:

  >>> from pyhdf.HDF import *
  >>> from pyhdf.V import *
  >>> f = HDF('test.hdf', HC.WRITE) # Open file 'test.hdf' in write mode
  >>> v = f.vgstart()            # init vgroup interface
  >>> vg = v.attach('vtest', 1)  # attach vdata 'vtest' in write mode
  >>> vg.version = '1.0'         # create vdata attribute 'version',
                                 # setting it to string '1.0'
  >>> print(vg.version)           # print attribute value
  >>> vg.range = (-10, 15)       # create attribute 'range', setting
                                 # it to the pair of ints (-10, 15)
  >>> print(vg.range)             # print attribute value
  >>> vg.detach()                # "close" the vdata
  >>> v.end()                    # terminate the vdata interface
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
starts with an underscore ('_') is either a "predefined" HDF attribute
(see below) or a standard python attribute. Otherwise, the attribute
is handled as an HDF attribute. Also, HDF attributes are not stored inside
the object dictionnary: the python dir() function will not list them.

Attribute values can be updated, but it is illegal to try to change the
value type, or the attribute order (number of values). This is important
for attributes holding string values. An attribute initialized with an
'n' character string is simply a character attribute of order 'n' (eg a
character array of length 'n'). If 'vg' is a vgroup and we initialize its
'a1' attribute as 'vg.a1 = "abcdef"', then a subsequent update attempt
like 'vg.a1 = "12"' will fail, because we then try to change the order
of the attribute (from 6 to 2). It is mandatory to keep the length of string
attributes constant.

Predefined attributes
---------------------
The VG class supports predefined attributes to get (and occasionnaly set)
attribute values easily using the usual python "dot notation", without
having to call a class method. The names of predefined attributes all start
with an underscore ('_').

In the following table, the RW column holds an X if the attribute
is read/write.

  VG predefined attributes

    =========== === =========================== ===================
    name        RW  description                 C library routine
    =========== === =========================== ===================
    _class      X   class name                  Vgetclass/Vsetclass
    _name       X   vgroup name                 Vgetname/Vsetname
    _nattrs         number of vgroup attributes Vnattrs
    _nmembers       number of vgroup members    Vntagrefs
    _refnum         vgroup reference number     VQueryref
    _tag            vgroup tag                  VQuerytag
    _version        vgroup version number       Vgetversion
    =========== === =========================== ===================


Programming models
------------------

Creating and initializing a vgroup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following program shows how to create and initialize a vgroup inside
an HDF file. It can serve as a model for any program wanting to create
a vgroup::

    from pyhdf.HDF import *
    from pyhdf.V   import *
    from pyhdf.VS  import *
    from pyhdf.SD  import *

    def vdatacreate(vs, name):

        # Create vdata and define its structure
        vd = vs.create(name,
                   (('partid',HC.CHAR8, 5),       # 5 char string
                    ('description',HC.CHAR8, 10), # 10 char string field
                    ('qty',HC.INT16, 1),          # 1 16 bit int field
                    ('wght',HC.FLOAT32, 1),       # 1 32 bit float
                    ('price',HC.FLOAT32,1)        # 1 32 bit float
                   ))

        # Store records
        vd.write((('Q1234', 'bolt',12, 0.01, 0.05),   # record 1
                  ('B5432', 'brush', 10, 0.4, 4.25),  # record 2
                  ('S7613', 'scissor', 2, 0.2, 3.75)  # record 3
                  ))
        # "close" vdata
        vd.detach()

    def sdscreate(sd, name):

        # Create a simple 3x3 float array.
        sds = sd.create(name, SDC.FLOAT32, (3,3))
        # Initialize array
        sds[:] = ((0,1,2),(3,4,5),(6,7,8))
        # "close" dataset.
        sds.endaccess()

    # Create HDF file
    filename = 'inventory.hdf'
    hdf = HDF(filename, HC.WRITE|HC.CREATE)

    # Initialize the SD, V and VS interfaces on the file.
    sd = SD(filename, SDC.WRITE)  # SD interface
    vs = hdf.vstart()             # vdata interface
    v  = hdf.vgstart()            # vgroup interface

    # Create vdata named 'INVENTORY'.
    vdatacreate(vs, 'INVENTORY')
    # Create dataset named "ARR_3x3"
    sdscreate(sd, 'ARR_3x3')

    # Attach the vdata and the dataset.
    vd = vs.attach('INVENTORY')
    sds = sd.select('ARR_3x3')

    # Create vgroup named 'TOTAL'.
    vg = v.create('TOTAL')

    # Add vdata to the vgroup
    vg.insert(vd)
    # We could also have written this:
    # vgroup.add(vd._tag, vd._refnum)
    # or this:
    # vgroup.add(HC.DFTAG_VH, vd._refnum)

    # Add dataset to the vgroup
    vg.add(HC.DFTAG_NDG, sds.ref())

    # Close vgroup, vdata and dataset.
    vg.detach()       # vgroup
    vd.detach()       # vdata
    sds.endaccess()   # dataset

    # Terminate V, VS and SD interfaces.
    v.end()           # V interface
    vs.end()          # VS interface
    sd.end()          # SD interface

    # Close HDF file.
    hdf.close()

The program starts by defining two functions vdatacreate() and sdscreate(),
which will serve to create the vdata and dataset objects we need. Those
functions are not essential to the example. They simply help to make the
example self-contained. Refer to the VS and SD module documentation for
additional explanations about how these functions work.

After opening the HDF file in write mode, the SD, V and VS interfaces are
initialized on the file. Next vdatacreate() is called to create a new vdata
named 'INVENTORY' on the VS instance, and sdscreate() to create a new
dataset named 'ARR_3x3' on the SD instance. This is done so that we have a
vdata and a dataset to play with.

The vdata and the dataset are then attached ("opened"). The create()
method of the V instance is then called to create a new vgroup named
'TOTAL'. The vgroup is then populated by calling its insert() method to add
the vdata 'INVENTORY', and its add() method to add the 'ARR_3x3' dataset.
Note that insert() is just a commodity method that simplifies adding a
vdata or a vgroup to a vgroup, avoiding the need to pass an object tag and
reference number. There is no such commodity method for adding a dataset
to a vgroup. The dataset must be added by specifying its tag and reference
number. Note that the tags to be used are defined inside the HDF module as
constants of the HC class: DFTAG_NDG for a dataset, DFTAG_VG for a vgroup,
DFTAG_VH for a vdata.

The program ends by detaching ("closing") the HDF objects created above,
terminating the three interfaces initialized, and closing the HDF file.


Reading a vgroup
^^^^^^^^^^^^^^^^
The following program shows the contents of the vgroups contained inside
any HDF file::

    from pyhdf.HDF import *
    from pyhdf.V   import *
    from pyhdf.VS  import *
    from pyhdf.SD  import *

    import sys

    def describevg(refnum):

        # Describe the vgroup with the given refnum.

        # Open vgroup in read mode.
        vg = v.attach(refnum)
        print "----------------"
        print "name:", vg._name, "class:",vg._class, "tag,ref:",
        print vg._tag, vg._refnum

        # Show the number of members of each main object type.
        print "members: ", vg._nmembers,
        print "datasets:", vg.nrefs(HC.DFTAG_NDG),
        print "vdatas:  ", vg.nrefs(HC.DFTAG_VH),
        print "vgroups: ", vg.nrefs(HC.DFTAG_VG)

        # Read the contents of the vgroup.
        members = vg.tagrefs()

        # Display info about each member.
        index = -1
        for tag, ref in members:
            index += 1
            print "member index", index
            # Vdata tag
            if tag == HC.DFTAG_VH:
                vd = vs.attach(ref)
                nrecs, intmode, fields, size, name = vd.inquire()
                print "  vdata:",name, "tag,ref:",tag, ref
                print "    fields:",fields
                print "    nrecs:",nrecs
                vd.detach()

            # SDS tag
            elif tag == HC.DFTAG_NDG:
                sds = sd.select(sd.reftoindex(ref))
                name, rank, dims, type, nattrs = sds.info()
                print "  dataset:",name, "tag,ref:", tag, ref
                print "    dims:",dims
                print "    type:",type
                sds.endaccess()

            # VS tag
            elif tag == HC.DFTAG_VG:
                vg0 = v.attach(ref)
                print "  vgroup:", vg0._name, "tag,ref:", tag, ref
                vg0.detach()

            # Unhandled tag
            else:
                print "unhandled tag,ref",tag,ref

        # Close vgroup
        vg.detach()

    # Open HDF file in readonly mode.
    filename = sys.argv[1]
    hdf = HDF(filename)

    # Initialize the SD, V and VS interfaces on the file.
    sd = SD(filename)
    vs = hdf.vstart()
    v  = hdf.vgstart()

    # Scan all vgroups in the file.
    ref = -1
    while 1:
        try:
            ref = v.getid(ref)
        except HDF4Error,msg:    # no more vgroup
            break
        describevg(ref)

    # Terminate V, VS and SD interfaces.
    v.end()
    vs.end()
    sd.end()

    # Close HDF file.
    hdf.close()

The program starts by defining function describevg(), which is passed the
reference number of the vgroup to display. The function assumes that the
SD, VS and V interfaces have been previously initialized.

The function starts by attaching ("opening") the vgroup, and displaying
its name, class, tag and reference number. The number of members of the
three most important object types is then displayed, by calling the nrefs()
method with the predefined tags found inside the HDF.HC class.

The tagrefs() method is then called to get a list of all the vgroup members,
each member being identified by its tag and reference number. A 'for'
statement is entered to loop over each element of this list. The tag is
tested against the known values defined in the HDF.HC class: the outcome of
this test indicates how to process the member object.

A DFTAG_VH tag indicates we deal with a vdata. The vdata is attached, its
inquire() method called to display info about it, and the vdata is detached.
In the case of a DFTAG_NFG, we are facing a dataset. The dataset is
selected, info is obtained by calling the dataset info() method, and the
dataset is released. A DFTAG_VG indicates that the member is a vgroup. We
attach it, print its name, tag and reference number, then detach the
member vgroup. A warning is finally displayed if we hit upon a member of
an unknown type.

The function releases the vgroup just displayed and returns.

The main program starts by opening in readonly mode the HDF file passed
as argument on the command line. The SD, VS and V interfaces are
initialized, and the corresponding class instances are stored inside 'sd',
'vs' and 'v' global variables, respectively, for the use of the
describevg() function.

A while loop is then entered to access each vgroup in the file. A reference
number of -1 is passed on the first call to getid() to obtain the reference
number of the first vgroup. getid() returns a new reference number on each
subsequent call, and raises an exception when the last vgroup has been
retrieved. This exception is caught to break out of the loop, otherwise
describevg() is called to display the vgroup we have on hand.

Once the loop is over, the interfaces initialized before are terminated,
and the HDF file is closed.

You will notice that this program will display vgroups other than those
you have explicitly created. Those supplementary vgroups are created
by the HDF library for its own internal needs.

"""

import os, sys, types

from . import hdfext as _C

from .six.moves import xrange
from .HC import HC
from .VS import VD
from .error import HDF4Error, _checkErr

# List of names we want to be imported by an "from pyhdf.V import *"
# statement

__all__ = ['V', 'VG', 'VGAttr']

class V(object):
    """The V class implements the V (Vgroup) interface applied to an
    HDF file.
    To instantiate a V class, call the vgstart() method of an
    HDF instance. """

    def __init__(self, hinst):
        # Not to be called directly by the user.
        # A V object is instantiated using the vgstart()
        # method of an HDF instance.

        # Args:
        #    hinst    HDF instance
        # Returns:
        #    A V instance
        #
        # C library equivalent : Vstart (rather: Vinitialize)

        # Private attributes:
        #  _hdf_inst:       HDF instance

        # Note: Vstart is just a macro; use 'Vinitialize' instead
        # Note also thet the same C function is used to initialize the
        # VS interface.
        status = _C.Vinitialize(hinst._id)
        _checkErr('V', status, "cannot initialize V interface")
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
        """Close the V interface.

        Args::

          No argument

        Returns::

          None

        C library equivalent : Vend
                                                """

        # Note: Vend is just a macro; use 'Vfinish' instead
        # Note also the the same C function is used to end
        # the VS interface
        _checkErr('vend', _C.Vfinish(self._hdf_inst._id),
                  "cannot terminate V interface")
        self._hdf_inst = None

    def attach(self, num_name, write=0):
        """Open an existing vgroup given its name or its reference
        number, or create a new vgroup, returning a VG instance for
        that vgroup.

        Args::

          num_name      reference number or name of the vgroup to open,
                        or -1 to create a new vgroup; vcreate() can also
                        be called to create and name a new vgroup
          write         set to non-zero to open the vgroup in write mode
                        and to 0 to open it in readonly mode (default)

        Returns::

          VG instance for the vgroup

        An exception is raised if an attempt is made to open
        a non-existent vgroup.

        C library equivalent : Vattach
                                                """

        if isinstance(num_name, bytes):
            num = self.find(num_name)
        else:
            num = num_name
        vg_id = _C.Vattach(self._hdf_inst._id, num,
                           write and 'w' or 'r')
        _checkErr('vattach', vg_id, "cannot attach Vgroup")
        return VG(self, vg_id)

    def create(self, name):
        """Create a new vgroup, and assign it a name.

        Args::

          name   name to assign to the new vgroup

        Returns::

          VG instance for the new vgroup

        A create(name) call is equivalent to an attach(-1, 1) call,
        followed by a call to the setname(name) method of the instance.

        C library equivalent : no equivalent
                                                    """

        vg = self.attach(-1, 1)
        vg._name = name
        return vg

    def find(self, name):
        """Find a vgroup given its name, returning its reference
        number if found.

        Args::

          name     name of the vgroup to find

        Returns::

          vgroup reference number

        An exception is raised if the vgroup is not found.

        C library equivalent: Vfind
                                                  """

        refnum = _C.Vfind(self._hdf_inst._id, name)
        if not refnum:
            raise HDF4Error("vgroup not found")
        return refnum

    def findclass(self, name):
        """Find a vgroup given its class name, returning its reference
        number if found.

        Args::

          name     class name of the vgroup to find

        Returns::

          vgroup reference number

        An exception is raised if the vgroup is not found.

        C library equivalent: Vfind
                                                  """

        refnum = _C.Vfindclass(self._hdf_inst._id, name)
        if not refnum:
            raise HDF4Error("vgroup not found")
        return refnum

    def delete(self, num_name):
        """Delete from the HDF file the vgroup identified by its
        reference number or its name.

        Args::

          num_name    either the reference number or the name of
                      the vgroup to delete

        Returns::

          None

        C library equivalent : Vdelete
                                             """

        try:
            vg = self.attach(num_name, 1)
        except HDF4Error as msg:
            raise HDF4Error("delete: no such vgroup")

        # ATTENTION: The HDF documentation says that the vgroup_id
        #            is passed to Vdelete(). This is wrong.
        #            The vgroup reference number must instead be passed.
        refnum = vg._refnum
        vg.detach()
        _checkErr('delete', _C.Vdelete(self._hdf_inst._id, refnum),
                  "error deleting vgroup")

    def getid(self, ref):
        """Obtain the reference number of the vgroup following the
        vgroup with the given reference number .

        Args::

          ref    reference number of the vgroup after which to search;
                 set to -1 to start the search at the start of
                 the HDF file

        Returns::

          reference number of the vgroup past the one identified by 'ref'

        An exception is raised if the end of the vgroup is reached.

        C library equivalent : Vgetid
                                                    """

        num = _C.Vgetid(self._hdf_inst._id, ref)
        _checkErr('getid', num, "bad arguments or last vgroup reached")
        return num


class VG(object):
    """The VG class encapsulates the functionnality of a vgroup.
    To instantiate a VG class, call the attach() or create() methods
    of a V class instance."""

    def __init__(self, vinst, id):
        # This construtor is not intended to be called directly
        # by the user program. The attach() method of an
        # V class instance should be called instead.

        # Arg:
        #  vinst       V instance from which the call is made
        #  id          vgroup identifier

        # Private attributes:
        #  _v_inst    V instance to which the vdata belongs
        #  _id        vgroup identifier

        self._v_inst = vinst
        self._id = id

    def __del__(self):
        """Delete the instance, first calling the detach() method
        if not already done.          """

        try:
            if self._id:
                self.detach()
        except:
            pass

    def __getattr__(self, name):
        """Some vgroup properties can be queried/set through the following
        attributes. Their names all start with an "_" to avoid
        clashes with user-defined attributes. Most are read-only.
        Only the _class and _name attributes can be modified.

        Name       RO  Description              C library routine
        -----      --  -----------------        -----------------
        _class         class name               Vgetclass/Vsetlass
        _name          name                     Vgetname/Vsetname
        _nattrs    X   number of attributes     Vnattrs
        _nmembers  X   number of vgroup members Vntagrefs
        _refnum    X   reference number         VQueryref
        _tag       X   tag                      VQuerytag
        _version   X   version number           Vgetversion

                                                         """

        # NOTE: python will call this method only if the attribute
        #       is not found in the object dictionnary.

        # Check for a user defined attribute first.
        att = self.attr(name)
        if att._index is not None:   # Then the attribute exists
            return att.get()

        # Check for a predefined attribute
        if name == "_class":
            status, nm = _C.Vgetclass(self._id)
            _checkErr('_class', status, 'cannot get vgroup class')
            return nm

        elif name == "_name":
            status, nm = _C.Vgetname(self._id)
            _checkErr('_name', status, 'cannot get vgroup name')
            return nm

        elif name == "_nattrs":
            n = _C.Vnattrs(self._id)
            _checkErr('_nattrs', n, 'cannot get number of attributes')
            return n

        elif name == "_nmembers":
            n = _C.Vntagrefs(self._id)
            _checkErr('refnum', n, 'cannot get vgroup number of members')
            return n

        elif name == "_refnum":
            n = _C.VQueryref(self._id)
            _checkErr('refnum', n, 'cannot get vgroup reference number')
            return n

        elif name == "_tag":
            n = _C.VQuerytag(self._id)
            _checkErr('_tag', n, 'cannot get vgroup tag')
            return n

        elif name == "_version":
            n = _C.Vgetversion(self._id)
            _checkErr('_tag', n, 'cannot get vgroup version')
            return n

        else:
            raise AttributeError

    def __setattr__(self, name, value):

        # A name starting with an underscore will be treated as
        # a standard python attribute, and as an HDF attribute
        # otherwise.

        # Forbid assigning to readonly attributes
        if name in ["_nattrs", "_nmembers", "_refnum", "_tag", "_version"]:
            raise AttributeError("%s: read-only attribute" % name)

        # Read-write predefined attributes
        elif name == "_class":
            _checkErr(name, _C.Vsetclass(self._id, value),
                      'cannot set _class property')

        elif name == "_name":
            _checkErr(name, _C.Vsetname(self._id, value),
                      'cannot set _name property')

        # Try to set a user-defined attribute.
        else:
            _setattr(self, name, value)

    def insert(self, inst):
        """Insert a vdata or a vgroup in the vgroup.

        Args::

          inst  vdata or vgroup instance to add

        Returns::

          index of the inserted vdata or vgroup (0 based)

        C library equivalent : Vinsert
                                                  """

        if isinstance(inst, VD):
            id = inst._id
        elif isinstance(inst, VG):
            id = inst._id
        else:
            raise HDF4Error("insrt: bad argument")

        index = _C.Vinsert(self._id, id)
        _checkErr('insert', index, "cannot insert in vgroup")
        return index

    def add(self, tag, ref):
        """Add to the vgroup an object identified by its tag and
        reference number.

        Args::

          tag       tag of the object to add
          ref       reference number of the object to add

        Returns::

          total number of objects in the vgroup after the addition

        C library equivalent : Vaddtagref
                                              """

        n = _C.Vaddtagref(self._id, tag, ref)
        _checkErr('addtagref', n, 'invalid arguments')
        return n

    def delete(self, tag, ref):
        """Delete from the vgroup the member identified by its tag
        and reference number.

        Args::

          tag    tag of the member to delete
          ref    reference number of the member to delete

        Returns::

          None

        Only the link of the member with the vgroup is deleted.
        The member object is not deleted.

        C library equivalent : Vdeletatagref
                                                  """

        _checkErr('delete', _C.Vdeletetagref(self._id, tag, ref),
                  "error deleting member")

    def detach(self):
        """Terminate access to the vgroup.

        Args::

          no argument

        Returns::

          None

        C library equivalent : Vdetach
                                              """

        _checkErr('detach', _C.Vdetach(self._id),
                  "cannot detach vgroup")
        self._id = None

    def tagref(self, index):
        """Get the tag and reference number of a vgroup member,
        given the index number of that member.

        Args::

          index   member index (0 based)

        Returns::

          2-element tuple:
            - member tag
            - member reference number

        C library equivalent : Vgettagref
                                                  """

        status, tag, ref = _C.Vgettagref(self._id, index)
        _checkErr('tagref', status, "illegal arguments")
        return tag, ref

    def tagrefs(self):
        """Get the tags and reference numbers of all the vgroup
        members.

        Args::

          no argument

        Returns::

          list of (tag,ref) tuples, one for each vgroup member

        C library equivalent : Vgettagrefs
                                                      """
        n = self._nmembers
        ret = []
        if n:
            tags = _C.array_int32(n)
            refs = _C.array_int32(n)
            k = _C.Vgettagrefs(self._id, tags, refs, n)
            _checkErr('tagrefs', k, "error getting tags and refs")
            for m in xrange(k):
                ret.append((tags[m], refs[m]))

        return ret

    def inqtagref(self, tag, ref):
        """Determines if an object identified by its tag and reference
        number belongs to the vgroup.

        Args::

          tag      tag of the object to check
          ref      reference number of the object to check

        Returns::

          False (0) if the object does not belong to the vgroup,
          True  (1) otherwise

        C library equivalent : Vinqtagref
                                             """

        return _C.Vinqtagref(self._id, tag, ref)

    def nrefs(self, tag):
        """Determine the number of tags of a given type in a vgroup.

        Args::

          tag    tag type to look for in the vgroup

        Returns::

          number of members identified by this tag type

        C library equivalent : Vnrefs
                                              """

        n = _C.Vnrefs(self._id, tag)
        _checkErr('nrefs', n, "bad arguments")
        return n

    def isvg(self, ref):
        """Determines if the member of a vgoup is a vgroup.

        Args::

          ref      reference number of the member to check

        Returns::

          False (0) if the member is not a vgroup
          True  (1) otherwise

        C library equivalent : Visvg
                                             """

        return _C.Visvg(self._id, ref)

    def isvs(self, ref):
        """Determines if the member of a vgoup is a vdata.

        Args::

          ref      reference number of the member to check

        Returns::

          False (0) if the member is not a vdata,
          True  (1) otherwise

        C library equivalent : Visvs
                                             """

        return _C.Visvs(self._id, ref)

    def attr(self, name_index):
        """Create a VGAttr instance representing a vgroup attribute.

        Args::

          name_index  attribute name or attribute index number; if a
                      name is given the attribute may not exist; in that
                      case, it will be created when the VGAttr
                      instance set() method is called

        Returns::

          VGAttr instance for the attribute. Call the methods of this
          class to query, read or set the attribute.

        C library equivalent : no equivalent

                                """

        return VGAttr(self, name_index)

    def attrinfo(self):
        """Return info about all the vgroup attributes.

        Args::

          no argument

        Returns::

          dictionnary describing each vgroup attribute; for each attribute,
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


    def findattr(self, name):
        """Search the vgroup for a given attribute.

        Args::

          name    attribute name

        Returns::

          if found, VGAttr instance describing the attribute
          None otherwise

         C library equivalent : Vfindattr
                                                  """

        try:
            att = self.attr(name)
            if att._index is None:
                att = None
        except HDF4Error:
            att = None
        return att


class VGAttr(object):
    """The VGAttr class encapsulates methods used to set and query
    attributes defined on a vgroup. To create an instance of this class,
    call the attr() method of a VG class.    """

    def __init__(self, obj, name_or_index):
        # This constructor should not be called directly by the user
        # program. The attr() method of a VG class must be called to
        # instantiate this class.

        # Args:
        #  obj            VG instance to which the attribute belongs
        #  name_or_index  name or index of the attribute; if a name is
        #                 given, an attribute with that name will be
        #                 searched, if not found, a new index number will
        #                 be generated

        # Private attributes:
        #  _v_inst        V instance
        #  _index         attribute index or None
        #  _name          attribute name or None

        self._v_inst = obj
        # Name is given. Attribute may exist or not.
        if isinstance(name_or_index, type('')):
            self._name = name_or_index
            self._index = _C.Vfindattr(self._v_inst._id, self._name)
            if self._index < 0:
                self._index = None
        # Index is given. Attribute must exist.
        else:
            self._index = name_or_index
            status, self._name, data_type, n_values, size = \
                    _C.Vattrinfo(self._v_inst._id, self._index)
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

        Note that a vgroup attribute can also be queried like a standard
        python class attribute by applying the usual "dot notation" to a
        VG instance.

        C library equivalent : Vgetattr

                                                """
        # Make sure the attribute exists.
        if self._index is None:
            raise HDF4Error("non existent attribute")
        # Obtain attribute type and the number of values.
        status, aName, data_type, n_values, size = \
                    _C.Vattrinfo(self._v_inst._id, self._index)
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

        status = _C.Vgetattr(self._v_inst._id, self._index, buf)
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

        Note that a vgroup attribute can also be set like a standard
        python class attribute by applying the usual "dot notation" to a
        VG instance.

        C library equivalent : Vsetattr

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
        status = _C.Vsetattr(self._v_inst._id, self._name, data_type,
                             n_values, buf)
        _checkErr('set', status, 'cannot execute')
        # Update the attribute index
        self._index = _C.Vfindattr(self._v_inst._id, self._name)
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

        C library equivalent : Vattrinfo
                                                           """

        # Make sure the attribute exists.
        if self._index is None:
            raise HDF4Error("non existent attribute")

        status, name, type, order, size = \
                _C.Vattrinfo(self._v_inst._id, self._index)
        _checkErr('info', status, "execution error")
        return name, type, order, size


###########################
# Support functions
###########################

def _setattr(obj, name, value):
    # Called by the __setattr__ method of the VG object.
    #
    #  obj   instance on which the attribute is set
    #  name  attribute name
    #  value attribute value

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
