# import os
# import sys

# class SuppressPrint:
#   def __enter__(self):
#     self._original_stdout = sys.stdout
#     sys.stdout = open(os.devnull, 'w')

#   def __exit__(self, exc_type, exc_val, exc_tb):
#     sys.stdout.close()
#     sys.stdout = self._original_stdout


from contextlib import contextmanager
import sys, os

@contextmanager
def SuppressPrint():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout