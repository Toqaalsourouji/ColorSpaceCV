#!/usr/bin/env python3
"""
Wrapper so you can do:

    python -m iconverter INPUT SRC DST -o OUTPUT

It simply exposes the converter module’s functions.
"""

from converter import convert_color_space, main

__all__ = ['convert_color_space', 'main']
