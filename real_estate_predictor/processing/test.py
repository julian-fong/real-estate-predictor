import os
import pathlib
def test():
    print(__file__)
    print(pathlib.Path(__file__).parent.parent.absolute().joinpath('storage', 'datasets'))