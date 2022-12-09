# -*- coding: utf-8 -*-

# Copyright 2020 Jungil Kong
#  MIT License (https://opensource.org/licenses/MIT)

"""."""
'''
Copied from https://github.com/jik876/hifi-gan/blob/master/utils.py
'''

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)