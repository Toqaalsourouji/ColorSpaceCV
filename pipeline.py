#!/usr/bin/env python3
import os
import sys
import glob
import argparse

import cv2
import numpy as np

from channelsdetect import detect_channels
from converter import convert_color_space, main
from color_space_detector import detect_color_space_by_mse



BANNER = r"""
===============================
       COLOR   SPACE   TOOL
===============================

USAGE:

  1) Convert an image between color spaces:
     python pipeline.py convert <INPUT_IMAGE_PATH> <SRC_SPACE> <DST_SPACE> -o <OUTPUT_PATH>
     e.g. python pipeline.py convert RGB1.jpg RGB HSV -o Output_hsv.png

  2) Detect the color space of a .npy dump:
     python pipeline.py detect <PATH_TO_NPY>
     e.g. python pipeline.py detect YUV1.npy

  3) Show the channel count of an image:
     The channel count will be printed either while converting or detecting.
  

"""



# Valid spaces for conversion (must match iconverter.SUPPORTED_SPACES)
VALID_SPACES = ['RGB', 'BGR', 'HSV', 'LAB', 'GRAY', 'HLS', 'YUV', 'XYZ', 'RGBA', 'BGRA']

def handle_convert(args):
    """Convert an image from src_space → dst_space, preserving alpha."""
    # Load input (preserves alpha if present)
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ Error: cannot open '{args.input}'", file=sys.stderr)
        sys.exit(1)

    # Do the conversion
    try:
        out = convert_color_space(img, args.src_space, args.dst_space)
    except ValueError as e:
        print(f"❌ Conversion error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save result
    if not cv2.imwrite(args.output, out):
        print(f"❌ Error: cannot save '{args.output}'", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Converted {args.src_space} → {args.dst_space}, saved to {args.output}")

def handle_detect(args):
    """Detect the color space of a .npy array, or at least print channel count."""
    path = args.input
    ext = os.path.splitext(path)[1].lower()

    if ext == '.npy':
        # Use the MSE‐based detector
        try:
            space = detect_color_space_by_mse(path)
            print(f"✅ Detected color space: {space}")
        except Exception as e:
            print(f"❌ Detection error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Fallback: just report channel count
        try:
            ch = detect_channels(path)
            print(f"ℹ️  Channel count: {ch}")
            if ch == 1:
                print(" → Likely GRAY.")
            elif ch == 4:
                print(" → RGBA or BGRA (alpha present).")
            else:
                print(" → 3 channels: try converting to .npy first for full detection.")
        except Exception as e:
            print(f"❌ Channel detection error: {e}", file=sys.stderr)
            sys.exit(1)

def main():
    print(BANNER)
    parser = argparse.ArgumentParser(
        prog="color_tool.py",
        description="🎨 Color Space Converter & Detector"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) Convert sub-command
    p_conv = sub.add_parser("convert", help="Convert image between color spaces")
    p_conv.add_argument("input",    help="Path to input image (png/jpg/etc.)")
    p_conv.add_argument("src_space", choices=VALID_SPACES,
                        help="Source color space")
    p_conv.add_argument("dst_space", choices=VALID_SPACES,
                        help="Target color space")
    p_conv.add_argument("-o", "--output", required=True,
                        help="Where to save the converted image")
    p_conv.set_defaults(func=handle_convert)

    # 2) Detect sub-command
    p_det = sub.add_parser("detect", help="Detect color space of an .npy array")
    p_det.add_argument("input", help="Path to .npy (or image) to analyze")
    p_det.set_defaults(func=handle_detect)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
