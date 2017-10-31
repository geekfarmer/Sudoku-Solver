# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 00:42:01 2017

@author: anura
"""

__author__ = 'saideeptalari'
import numpy as np

class SolveSudoku(object):
    def __init__(self,grid):
        self.grid = grid

    def _check_row(self,digit,row):
        return digit not in self.grid[row,:]

    def _check_column(self,digit,column):
        return digit not in self.grid[:,column]

    def _check_box(self,digit,row,col):
        rc = (row/3)*3
        cc = (col/3)*3
        return digit not in self.grid[rc:rc+3,cc:cc+3]

    def _check_safe(self,digit,row,col):
        return self._check_box(digit,row,col) \
               and self._check_row(digit,row) \
               and self._check_column(digit,col)

    def _find_empty(self,stop):
        if 0 in self.grid:
            stop[0] = np.where(self.grid==0)[0][0]
            stop[1] = np.where(self.grid==0)[1][0]
            return True
        else:
            return False

    def _solve(self):
        stop = [0,0]
        if not self._find_empty(stop):
            return True
        for digit in np.arange(1,10):
            row = stop[0]
            col = stop[1]
            if self._check_safe(digit,row,col):
                self.grid[row,col] = digit
                if self._solve():
                    return True
                self.grid[row,col]=0
        return False

    def solve(self):
        if self._solve():
            return self.grid
        raise("Sudoku can not be solved")
