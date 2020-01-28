"""
Copyright (c) <2019> <CEI Lab, Duke University>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

class ops_gen_v3:
    def __init__(self):
        self.ops_def = np.asarray([{'type': 'SeparableConv', 'filters': 16, 'kernel_size': 3, 'strides': 1},
                        {'type': 'Convolutional', 'filters': 16, 'kernel_size': 1, 'strides': 1},])

    def seed_ops(self):
        id = np.random.randint(0, len(self.ops_def))
        print(id)
        return self.ops_def[id]

    def seed_group_ops(self, n):
        """
        Seed a group of ops
        :param n:
        :return:
        """
        rand_list = []
        m = len(self.ops_def)
        for i in range(n):
            rand_list.append(i % m)
        np.random.shuffle(rand_list)
        rand_list = np.asarray(rand_list, np.int)
        print(self.ops_def[rand_list])
        return self.ops_def[rand_list]


