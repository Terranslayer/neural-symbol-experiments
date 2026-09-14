# -*- coding: utf-8 -*-

class WorkingMemory:
    """K 个槽位；分配 D 给字典，R 给寄存；强制 D+R=K。"""

    def __init__(self, K: int):
        if not isinstance(K, int) or K <= 1:
            raise ValueError("K must be int > 1")
        self.K = K
        self.D = None
        self.R = None

    def allocate(self, D: int):
        if not isinstance(D, int):
            raise TypeError("D must be int")
        if D <= 0 or D >= self.K:
            raise ValueError("0 < D < K must hold")
        self.D = D
        self.R = self.K - D
        return self.D, self.R
