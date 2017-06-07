#Copyright ReportLab Europe Ltd. 2000-2017
#see license.txt for license details
__doc__="""The Reportlab PDF generation library."""
Version = "3.4.0"
__version__=Version
__date__='20170307'

import sys, os

if sys.version_info[0:2]!=(2, 7) and sys.version_info<(3, 3):
    raise ImportError("""reportlab requires Python 2.7+ or 3.3+; 3.0-3.2 are not supported.""")

#define these early in reportlab's life
isPy3 = sys.version_info[0]==3
if isPy3:
    def cmp(a,b):
        return -1 if a<b else (1 if a>b else 0)

    import builtins
    builtins.cmp = cmp
    builtins.xrange = range
    del cmp, builtins
    def _fake_import(fn,name):
        from importlib import machinery
        m = machinery.SourceFileLoader(name,fn)
        try:
            sys.modules[name] = m.load_module(name)
        except FileNotFoundError:
            raise ImportError('file %s not found' % ascii(fn))
else:
    from future_builtins import ascii
    import __builtin__
    __builtin__.ascii = ascii
    del ascii, __builtin__
    def _fake_import(fn,name):
        if os.path.isfile(fn):
            import imp
            with open(fn,'rb') as f:
                sys.modules[name] = imp.load_source(name,fn,f)

#try to use dynamic modifications from
#reportlab.local_rl_mods.py
#reportlab_mods.py or ~/.reportlab_mods
try:
    import reportlab.local_rl_mods
except ImportError:
    pass

try:
    import reportlab_mods   #application specific modifications can be anywhere on python path
except ImportError:
    try:
        _fake_import(os.path.expanduser(os.path.join('~','.reportlab_mods')),'reportlab_mods')
    except (ImportError,KeyError):
        pass
