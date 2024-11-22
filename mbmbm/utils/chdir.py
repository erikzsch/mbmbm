import os
from pathlib import Path
from typing import Union


class ChDir:
    """change the directory temporarily using 'with' and jumps back afterwards"""

    def __init__(self, dir: Union[str, Path], create_on_not_exists=False):
        """create an object but NOT jumps into the direcotry (therefore: __enter__ must be used)"""
        self.dir = str(dir)
        self.create_on_not_exists = create_on_not_exists

    def __enter__(self):
        """changes into the defidne directory

        if 'create_on_not_exists=True' the directory is created, if not exist.
        """
        self.cwd = os.getcwd()
        if not os.path.exists(self.dir):
            if self.create_on_not_exists:
                os.makedirs(self.dir)
            else:
                raise Exception("cannot find directory {}".format(os.path.abspath(os.path.join(os.getcwd(), self.dir))))
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """moves back to old (saved) path"""
        os.chdir(self.cwd)
