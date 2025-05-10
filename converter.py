# converter.py
import cv2
import numpy as np

# Supported spaces (including BGR intermediate and alpha)
SUPPORTED_SPACES = [
    'RGB', 'BGR', 'HSV', 'HLS', 'LAB', 'GRAY', 'YCrCb', 'YUV', 'XYZ', 'RGBA', 'BGRA'
]

def _build_convert_flags():
    """
    Build a dict mapping (SRC, DST) → cv2 conversion flag
    by scanning cv2.COLOR_<SRC>2<DST> constants, splitting only on the
    first '2' to avoid too-many-values errors.
    """
    flags = {}
    for name in dir(cv2):
        if not name.startswith('COLOR_') or '2' not in name:
            continue
        flag_val = getattr(cv2, name)
        body = name[len('COLOR_'):]              # e.g. "HSV2BGR_FULL" or "BayerBG2BGR"
        parts = body.split('2', 1)               # only split once
        if len(parts) != 2:
            continue
        src, dst = parts
        flags[(src.upper(), dst.upper())] = flag_val
    return flags


CONVERT_FLAGS = _build_convert_flags()

def convert_color_space(image: np.ndarray, src_space: str, dst_space: str) -> np.ndarray:
    """
    Convert `image` from src_space to dst_space, preserving any alpha channels.
    """
    src = src_space.upper()
    dst = dst_space.upper()
    if src not in SUPPORTED_SPACES or dst not in SUPPORTED_SPACES:
        raise ValueError(f"Unsupported: {src}→{dst}. Supported: {SUPPORTED_SPACES}")

    # No-op
    if src == dst:
        return image.copy()

    # Separate alpha
    alpha = None
    if image.ndim == 3 and image.shape[2] > 3:
        base, alpha = image[..., :3], image[..., 3:]
    else:
        base = image

    # SRC → BGR
    if src != 'BGR':
        flag_in = CONVERT_FLAGS.get((src, 'BGR'))
        if flag_in is None:
            raise ValueError(f"No conversion path {src}→BGR")
        base = cv2.cvtColor(base, flag_in)

    # BGR → DST
    if dst != 'BGR':
        flag_out = CONVERT_FLAGS.get(('BGR', dst))
        if flag_out is None:
            raise ValueError(f"No conversion path BGR→{dst}")
        base = cv2.cvtColor(base, flag_out)

    # Reattach alpha
    if alpha is not None:
        if base.ndim == 2:
            base = base[:, :, np.newaxis]
        base = np.concatenate((base, alpha), axis=2)

    return base

def main():
    """
    CLI entry point:
        python converter.py INPUT SRC_SPACE DST_SPACE -o OUTPUT
    """
    import argparse, sys

    parser = argparse.ArgumentParser(prog='converter.py',
        description='Convert image between color spaces (preserves alpha).')
    parser.add_argument('input',    help='Path to input image')
    parser.add_argument('src_space',choices=SUPPORTED_SPACES, help='Source space')
    parser.add_argument('dst_space',choices=SUPPORTED_SPACES, help='Target space')
    parser.add_argument('-o','--output', required=True, help='Output path')
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: cannot load '{args.input}'", file=sys.stderr)
        sys.exit(1)

    try:
        out = convert_color_space(img, args.src_space, args.dst_space)
    except ValueError as e:
        print(f"Conversion error: {e}", file=sys.stderr)
        sys.exit(1)

    if not cv2.imwrite(args.output, out):
        print(f"Error: cannot save '{args.output}'", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {args.src_space} → {args.dst_space}, saved to {args.output}")

if __name__ == "__main__":
    main()