# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 00:59:14 2020

@author: Zhenzi Weng
"""

import os
from random import choice

def path_dir(path, dtype=".wav"):
    fname = os.listdir(path)
    fname = [x for x in fname if x.endswith(dtype)]
    fname.sort()
    out_path = [os.path.join(path, name) for name in fname]    
    file_path = choice(out_path)
    
    return file_path
