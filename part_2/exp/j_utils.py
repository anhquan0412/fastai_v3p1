# helpful and fast functions using minimal amount of library

import operator
from pathlib import Path
from IPython.core.debugger import set_trace
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
# from torch import tensor
from functools import partial
from collections import OrderedDict

# test units
def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')
    
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)
def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"


import re
_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name): # ThisThatName -> this_that
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


# -ify functions
from typing import *
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def setify(o): return o if isinstance(o,set) else set(listify(o))


def uniqueify(x, sort=False):
    # exclude duplicate values while maintaining original order
    res = list(OrderedDict.fromkeys(x).keys())
    if sort: res.sort()
    return res


class ListContainer(): # use numpy-styled indexing on any list of items
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        try: return self.items[idx]
        except TypeError:
            if isinstance(idx[0],bool):
                assert len(idx)==len(self) # bool mask
                return [o for m,o in zip(idx,self.items) if m]
            return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res

# image file I/O

import PIL,os,mimetypes
image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def _get_files(p, fs, extensions=None):
    # go through one directory and grab all img type files. Return full file path
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res

def get_files(path, extensions=None, recurse=False, include=None):
    # return the full path of files, recursively or not
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        # walk recursively in directories to find files
        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
            
            # we can change the list of directory we want os to 'walk' through next, using 'include'
            if include is not None and i==0: d[:] = [o for o in d if o in include]
            else:                            d[:] = [o for o in d if not o.startswith('.')]
                
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()] #get only file name from the path
        return _get_files(path, f, extensions)
                
