# init for netCDF4. package
# Docstring comes from extension module _netCDF4.
from ._netCDF4 import *
# Need explicit imports for names beginning with underscores
from ._netCDF4 import __doc__, __pdoc__
from ._netCDF4 import (__version__, __netcdf4libversion__, __hdf5libversion__,
                       __has_rename_grp__, __has_nc_inq_path__,
                       __has_nc_inq_format_extended__, __has_nc_open_mem__,
                       __has_cdf5_format__,__has_nc_par__)
__all__ =\
['Dataset','Variable','Dimension','Group','MFDataset','MFTime','CompoundType','VLType','date2num','num2date','date2index','stringtochar','chartostring','stringtoarr','getlibversion','EnumType']
