from PyQt5 import uic

with open('interface_new.py', 'w') as fout:
    uic.compileUi('Mock.ui', fout)