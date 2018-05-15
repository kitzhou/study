#!virtualenv tensorflow python3
#-*-coding: utf-8-*-
''''''
#just put some common things here, like approot
__title__ = ''
__author__ = 'zxx'
__mtime__ = '18-5-15'

import os

def getProjectRoot():
    return os.path.dirname(os.path.abspath(__file__))
