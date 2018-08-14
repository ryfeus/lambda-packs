#!/usr/bin/env python

import numpy as np
import os
import pyhdf.SD
import tempfile
from nose.tools import eq_
from pyhdf.SD import SDC

def test_long_varname():
    sds_name = 'a'*255

    _, path = tempfile.mkstemp(suffix='.hdf', prefix='pyhdf_')
    try:
        # create a file with a long variable name
        sd = pyhdf.SD.SD(path, SDC.WRITE|SDC.CREATE|SDC.TRUNC)
        sds = sd.create(sds_name, SDC.FLOAT32, (3,))
        sds[:] = range(10, 13)
        sds.endaccess()
        sd.end()

        # check we can read the variable name
        sd = pyhdf.SD.SD(path)
        sds = sd.select(sds_name)
        name, _, _, _, _ = sds.info()
        sds.endaccess()
        sd.end()
        eq_(sds_name, name)
    finally:
        os.unlink(path)
