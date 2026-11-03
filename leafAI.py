#!/usr/bin/env python3
"""Cross-platform leaf and tea-bud image analysis CLI."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

cv = None
np = None
features = None

LOGGER = logging.getLogger("leafAI")
__version__ = "1.1"
DEFAULT_HSV = (0, 160, 0, 60, 50, 255)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRICHOME_SMAX_VALUES = tuple(range(40, 100, 10))
TRICHOME_VMIN_VALUES = tuple(range(50, 201, 10))
SHAPE_HEADERS = ["perimeter", "area", "length", "width", "aspect_ratio",
                 "extent", "solidity", "equivalent_diameter", "circularity"]
TEXTURE_HEADERS = ["texture_contrast", "texture_dissimilarity",
                   "texture_homogeneity", "texture_correlation", "texture_asm",
                   "texture_energy", "texture_entropy"]
COLOR_HEADERS = [
    f"{space}_{channel}_{moment}"
    for space, channels in (("hsv", ("h_deg", "s_pct", "v_pct")),
                            ("rgb", ("r", "g", "b")),
                            ("lab", ("l_star", "a_star", "b_star")))
    for channel in channels
    for moment in ("mean", "std", "third_root")
]
PROGRAM_MODES = (
    "shape", "texture", "color", "features",
    "trichome-index", "trichome-masks", "hsv-plot",
    "mark", "split",
    "summarize", "distribution", "join", "correlation",
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    """Keep defaults while preserving grouped help text formatting."""

    def _get_help_string(self, action):
        text = action.help or ""
        if ("%(default)" not in text and action.option_strings
                and action.default not in (None, False, argparse.SUPPRESS)):
            text += " (default: %(default)s)"
        return text


def program_mode(value: str) -> str:
    if value not in PROGRAM_MODES:
        raise argparse.ArgumentTypeError(
            f"unknown mode {value!r}; run `leafAI.py --help` to see grouped modes"
        )
    return value


def load_dependencies() -> None:
    """Load heavy dependencies after argument parsing so --help always works."""
    global cv, np, features
    try:
        import cv2
        import numpy
        from scripts import Features
    except ImportError as exc:
        package = "opencv-python" if exc.name == "cv2" else exc.name
        raise RuntimeError(
            f"missing dependency: {package}; run "
            "`python -m pip install -r requirements.txt`"
        ) from exc
    cv, np, features = cv2, numpy, Features


def hsv_threshold(value: str) -> tuple[int, int, int, int, int, int]:
    """Parse Hmin,Hmax,Smin,Smax,Vmin,Vmax."""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV values must be integers") from exc
    if len(values) != 6:
        raise argparse.ArgumentTypeError("HSV needs exactly 6 comma-separated values")
    hmin, hmax, smin, smax, vmin, vmax = values
    if not (0 <= hmin <= hmax <= 179):
        raise argparse.ArgumentTypeError("H must satisfy 0 <= Hmin <= Hmax <= 179")
    if not (0 <= smin <= smax <= 255 and 0 <= vmin <= vmax <= 255):
        raise argparse.ArgumentTypeError("S and V must be in 0..255 (min <= max)")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("leafAI: quantitative image analysis for leaf and tea-bud "
                     "phenotyping."),
        formatter_class=HelpFormatter,
        add_help=False,
        epilog="""OPERATION MODES
  Feature extraction
    shape             Extract geometric and morphological descriptors
    texture           Extract multi-directional GLCM texture descriptors
    color             Extract HSV, RGB, and CIE Lab color moments
    features          Extract shape, texture, and color descriptors together

  Trichome analysis
    trichome-index    Quantify trichome pixels using an HSV threshold
    trichome-masks    Generate an HSV threshold mask series for calibration
    hsv-plot          Plot 2-D HSV distributions and a 3-D HSV surface

  Image preprocessing
    mark              Detect, number, and annotate foreground objects
    split             Detect, deskew, and crop objects from images

  Tabular analysis
    summarize         Aggregate replicate measurements by biological group
    distribution      Report descriptive statistics and plot a distribution
    join              Join one trait from two CSV files by sample ID
    correlation       Calculate Pearson correlation and plot linear regression

EXAMPLES
  python leafAI.py -p features -i leaf.png -o results/leaf
  python leafAI.py -p split -d images -o results/split
  python leafAI.py -p summarize -v values.csv -g groups.csv -o summary.csv

See README.md for input schemas, output definitions, and complete workflows.""",
    )
    parser._optionals.title = "General options"
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    parser.add_argument("--version", action="version", version=f"leafAI v{__version__}",
                        help="Show the program version and exit")
    parser.add_argument(
        "-p", "--program", required=True, type=program_mode, metavar="MODE",
        help="Operation mode; see the categorized list below",
    )
    parser.add_argument("-i", "--infile", type=Path, help="Input image file")
    parser.add_argument("-d", "--indir", type=Path, help="Input image directory")
    parser.add_argument("-o", "--out", type=Path, default=Path("output"),
                        help="Output file prefix; parent directories are created automatically")
    parser.add_argument("-H", "--hsv-thresh", dest="hsv_thresh",
                        type=hsv_threshold, default=DEFAULT_HSV,
                        metavar="HMIN,HMAX,SMIN,SMAX,VMIN,VMAX",
                        help="HSV interval used by trichome-index")
    parser.add_argument("--pattern", default="*",
                        help="File-name glob applied within --indir")
    parser.add_argument("--dpi", type=float, default=300.0,
                        help="Source image DPI used for physical-unit conversion")
    parser.add_argument("--debug", action="store_true", help="Display full exception tracebacks")
    image_group = parser.add_argument_group("Image processing and feature options")
    image_group.add_argument("--min-area", type=float, default=3000,
                             help="Minimum accepted contour area in pixels squared")
    image_group.add_argument("--max-area", type=float, default=700000,
                             help="Maximum accepted contour area in pixels squared")
    image_group.add_argument("--border", type=int, default=500,
                             help="Black border added before object splitting, in pixels")
    image_group.add_argument("--margin", type=int, default=50,
                             help="Additional crop margin around each object, in pixels")
    image_group.add_argument("--texture-levels", type=int, default=32,
                             choices=(8, 16, 32, 64, 128, 256),
                             help="Gray-level quantization used for GLCM extraction")
    stats_group = parser.add_argument_group("Tabular analysis options")
    stats_group.add_argument("-v", "--value", type=Path, help="Input measurement CSV")
    stats_group.add_argument("-g", "--group", type=Path, help="Sample-to-group mapping CSV")
    stats_group.add_argument("-t", "--trait", help="Trait or column name")
    stats_group.add_argument("-a", "--afile", type=Path, help="Left input CSV for join")
    stats_group.add_argument("-b", "--bfile", type=Path, help="Right input CSV for join")
    stats_group.add_argument("--left-trait", help="Trait in the left CSV; defaults to --trait")
    stats_group.add_argument("--right-trait", help="Trait in the right CSV; defaults to --trait")
    stats_group.add_argument("-c", "--columns",
                             help="Two comma-separated output column names for join")
    return parser


def collect_images(infile: Path | None, indir: Path | None, pattern: str) -> list[Path]:
    if infile is None and indir is None:
        raise ValueError("provide --infile or --indir")
    images: list[Path] = []
    if indir is not None:
        if not indir.is_dir():
            raise ValueError(f"input directory does not exist: {indir}")
        images.extend(path for path in sorted(indir.glob(pattern))
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if infile is not None:
        if not infile.is_file():
            raise ValueError(f"input file does not exist: {infile}")
        images.append(infile)
    unique = list(dict.fromkeys(path.resolve() for path in images))
    if not unique:
        raise ValueError("no supported images matched the supplied input")
    return unique


def read_image(path: Path) -> np.ndarray:
    """Read Unicode paths consistently on Windows, macOS and Linux."""
    try:
        image = cv.imdecode(np.fromfile(str(path), dtype=np.uint8), cv.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read image: {path}") from exc
    if image is None:
        raise ValueError(f"unsupported or damaged image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Use an encoder matching the file suffix and Unicode-safe I/O."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv.imencode(suffix, image)
    if not ok:
        raise ValueError(f"OpenCV cannot encode {suffix} images")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise ValueError(f"cannot write image: {path}") from exc


def per_image_prefix(output: Path, image: Path, multiple: bool) -> Path:
    return output.parent / f"{output.name}_{image.stem}" if multiple else output


def process_shape(images: Sequence[Path], output: Path, dpi: float) -> int:
    rows, failures = [], 0
    for path in images:
        try:
            gray, binary = features.preprocess(read_image(path))
            prefix = per_image_prefix(output, path, len(images) > 1)
            write_image(prefix.parent / f"{prefix.name}_gray.png", gray)
            write_image(prefix.parent / f"{prefix.name}_binary.png", binary)
            rows.append([path.stem, *features.shape_features(binary, dpi=dpi)])
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)
    target = output.parent / f"{output.name}.shape.csv"
    features.write_csv(target, rows, ["ID", *SHAPE_HEADERS])
    LOGGER.info("Wrote %s (%d rows)", target, len(rows))
    return failures


def process_features(images: Sequence[Path], output: Path, mode: str,
                     dpi: float, texture_levels: int) -> int:
    """Extract texture, color, or the combined shape/texture/color feature set."""
    rows, failures = [], 0
    for path in images:
        try:
            image = read_image(path)
            _, binary = features.preprocess(image)
            mask = features.foreground_mask(image, binary=binary)
            values = []
            if mode == "features":
                values.extend(features.shape_features(binary, dpi=dpi))
            if mode in ("texture", "features"):
                values.extend(features.texture_features(image, levels=texture_levels, mask=mask))
            if mode in ("color", "features"):
                values.extend(features.color_features(image, mask=mask))
            rows.append([path.stem, *values])
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)

    headers = []
    if mode == "features":
        headers.extend(SHAPE_HEADERS)
    if mode in ("texture", "features"):
        headers.extend(TEXTURE_HEADERS)
    if mode in ("color", "features"):
        headers.extend(COLOR_HEADERS)
    target = output.parent / f"{output.name}.{mode}.csv"
    features.write_csv(target, rows, ["ID", *headers])
    LOGGER.info("Wrote %s (%d rows, %d feature(s))", target, len(rows), len(headers))
    return failures


def process_hsv_plot(images: Sequence[Path], output: Path) -> int:
    try:
        from scripts import Plot as plot
    except ImportError as exc:
        raise RuntimeError("hsv-plot needs the plotting dependencies") from exc
    failures = 0
    for path in images:
        try:
            h, s, v = features.hsv_pixels(read_image(path))
            prefix = per_image_prefix(output, path, len(images) > 1)
            distribution = prefix.parent / f"{prefix.name}.hsv-distribution.png"
            surface = prefix.parent / f"{prefix.name}.hsv-surface.png"
            plot.hsv_pair_plot(h, s, v, "h", "s", "v", distribution)
            plot.hsv_surface_plot(h, s, v, surface)
            LOGGER.info("Wrote %s and %s", distribution, surface)
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)
    return failures


def process_masks(images: Sequence[Path], output: Path) -> int:
    failures = 0
    for path in images:
        try:
            image = read_image(path)
            prefix = per_image_prefix(output, path, len(images) > 1)
            for smax in TRICHOME_SMAX_VALUES:
                for vmin in TRICHOME_VMIN_VALUES:
                    bounds = (0, 160, 0, smax, vmin, 255)
                    mask = features.trichome_mask(image, 0, 0, vmin, 160, smax, 255)
                    _, _, index = features.trichome_index(image, 0, 0, vmin,
                                                          160, smax, 255)
                    annotated = mask.copy()
                    cv.putText(annotated, f"TI: {index:.2f}", (20, 30),
                               cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                               1, cv.LINE_AA)
                    tag = "_".join(map(str, bounds))
                    write_image(prefix.parent / f"{prefix.name}_mask_{tag}.png", annotated)
            mask_count = len(TRICHOME_SMAX_VALUES) * len(TRICHOME_VMIN_VALUES)
            LOGGER.info("Wrote %d masks for %s", mask_count, path.name)
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)
    return failures


def process_index(images: Sequence[Path], output: Path, hsv: Sequence[int]) -> int:
    rows, failures = [], 0
    hmin, hmax, smin, smax, vmin, vmax = hsv
    for path in images:
        try:
            values = features.trichome_index(read_image(path), hmin, smin, vmin,
                                              hmax, smax, vmax)
            rows.append([path.stem, *values])
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)
    target = output.parent / f"{output.name}.trichome.csv"
    features.write_csv(target, rows,
                       ["ID", "trichome_pixel", "bud_pixel", "trichome_index"])
    LOGGER.info("Wrote %s (%d rows)", target, len(rows))
    return failures


def output_file(output: Path, default_suffix: str) -> Path:
    """Treat output as an exact file when it has a suffix, otherwise as a prefix."""
    return output if output.suffix else output.parent / f"{output.name}{default_suffix}"


def process_mark(images: Sequence[Path], output: Path,
                 min_area: float | None, max_area: float | None) -> int:
    from scripts.ImageProcessing import annotate_objects

    failures = 0
    for path in images:
        try:
            annotated, threshold, count = annotate_objects(
                read_image(path), min_area if min_area is not None else 3000,
                max_area if max_area is not None else 700000,
            )
            prefix = per_image_prefix(output, path, len(images) > 1)
            target = prefix.parent / f"{prefix.name}_marker.png"
            write_image(target, annotated)
            LOGGER.info("Wrote %s (%d object(s), threshold=%d)", target, count, threshold)
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)
    return failures


def process_split(images: Sequence[Path], output: Path, min_area: float | None,
                  max_area: float | None, border: int, margin: int) -> int:
    from scripts.ImageProcessing import split_objects

    failures = 0
    for path in images:
        try:
            crops, threshold = split_objects(
                read_image(path), min_area if min_area is not None else 3000,
                max_area if max_area is not None else 700000,
                border, margin,
            )
            prefix = per_image_prefix(output, path, len(images) > 1)
            generated_name = re.compile(rf"{re.escape(prefix.name)}_R\d+\.png")
            for previous in prefix.parent.glob(f"{prefix.name}_R*.png"):
                if generated_name.fullmatch(previous.name):
                    previous.unlink()
            for number, crop in enumerate(crops, 1):
                write_image(prefix.parent / f"{prefix.name}_R{number}.png", crop)
            LOGGER.info("Wrote %d crop(s) for %s (threshold=%d)",
                        len(crops), path.name, threshold)
        except Exception as exc:
            failures += 1
            LOGGER.error("Failed %s: %s", path, exc)
            LOGGER.debug("Image failure", exc_info=True)
    return failures


def require_argument(value, flag: str):
    if value is None or value == "":
        raise ValueError(f"{flag} is required for this operation")
    return value


def process_summary(value: Path | None, group: Path | None, output: Path) -> int:
    from scripts.Statistics import grouped_summary

    target = output_file(output, ".summary.csv")
    count = grouped_summary(require_argument(value, "--value"),
                            require_argument(group, "--group"), target)
    LOGGER.info("Wrote %s (%d group(s))", target, count)
    return 0


def process_distribution(value: Path | None, trait: str | None, output: Path) -> int:
    from scripts.Statistics import describe, trait_values
    try:
        from scripts import Plot as plot
    except ImportError as exc:
        raise RuntimeError("distribution needs plotting dependencies") from exc

    _, values = trait_values(require_argument(value, "--value"),
                             require_argument(trait, "--trait"))
    stats = describe(values)
    LOGGER.info("Count: %d", stats["count"])
    LOGGER.info("Minimum: %.2f", stats["minimum"])
    LOGGER.info("Maximum: %.2f", stats["maximum"])
    LOGGER.info("Average: %.2f", stats["mean"])
    LOGGER.info("Standard deviation: %.2f", stats["std"])
    LOGGER.info("Coefficient of variation: %.4f", stats["cv"])
    target = output_file(output, ".distribution.png")
    plot.histogram(values, target)
    LOGGER.info("Wrote %s", target)
    return 0


def process_join(afile: Path | None, bfile: Path | None, trait: str | None,
                 left_trait: str | None, right_trait: str | None,
                 columns: str | None, output: Path) -> int:
    from scripts.Statistics import join_traits

    common = trait
    left_name = require_argument(left_trait or common, "--left-trait or --trait")
    right_name = require_argument(right_trait or common, "--right-trait or --trait")
    column_names = None
    if columns:
        parsed = tuple(item.strip() for item in columns.split(","))
        if len(parsed) != 2 or not all(parsed):
            raise ValueError("--columns needs exactly two comma-separated names")
        column_names = parsed
    target = output_file(output, ".joined.csv")
    count = join_traits(require_argument(afile, "--afile"),
                        require_argument(bfile, "--bfile"), target,
                        left_name, right_name, column_names)
    LOGGER.info("Wrote %s (%d matched ID(s))", target, count)
    return 0


def process_correlation(value: Path | None, output: Path) -> int:
    from scripts.Statistics import repeat_pairs
    try:
        import pandas as pd
        from scipy.stats import pearsonr
        from scripts import Plot as plot
    except ImportError as exc:
        raise RuntimeError("correlation needs pandas, scipy and plotting dependencies") from exc

    x_name, y_name, x, y = repeat_pairs(require_argument(value, "--value"))
    correlation, p_value = pearsonr(x, y)
    LOGGER.info("Pearson correlation: %.6g; p-value: %.6g", correlation, p_value)
    data = pd.DataFrame({x_name: x, y_name: y})
    target = output_file(output, ".correlation.png")
    plot.linear_regression(data, target, x_name, y_name)
    LOGGER.info("Wrote %s", target)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s: %(message)s")
    if args.dpi <= 0:
        parser.error("--dpi must be greater than zero")
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        operation = args.program
        image_operations = {"shape", "texture", "color", "features",
                            "hsv-plot", "trichome-masks", "trichome-index",
                            "mark", "split"}
        image_count = None
        if operation in image_operations:
            load_dependencies()
            images = collect_images(args.infile, args.indir, args.pattern)
            image_count = len(images)
            handlers = {
                "shape": lambda: process_shape(images, args.out, args.dpi),
                "texture": lambda: process_features(images, args.out, "texture", args.dpi,
                                                     args.texture_levels),
                "color": lambda: process_features(images, args.out, "color", args.dpi,
                                                   args.texture_levels),
                "features": lambda: process_features(images, args.out, "features", args.dpi,
                                                      args.texture_levels),
                "hsv-plot": lambda: process_hsv_plot(images, args.out),
                "trichome-masks": lambda: process_masks(images, args.out),
                "trichome-index": lambda: process_index(images, args.out, args.hsv_thresh),
                "mark": lambda: process_mark(images, args.out, args.min_area, args.max_area),
                "split": lambda: process_split(images, args.out, args.min_area, args.max_area,
                                                args.border, args.margin),
            }
            failures = handlers[operation]()
        elif operation == "summarize":
            failures = process_summary(args.value, args.group, args.out)
        elif operation == "distribution":
            failures = process_distribution(args.value, args.trait, args.out)
        elif operation == "join":
            failures = process_join(args.afile, args.bfile, args.trait, args.left_trait,
                                    args.right_trait, args.columns, args.out)
        else:
            failures = process_correlation(args.value, args.out)
    except (ValueError, RuntimeError, OSError) as exc:
        LOGGER.error("%s", exc)
        if args.debug:
            LOGGER.exception("Details")
        return 2
    if failures:
        LOGGER.warning("Completed with %d failed image(s)", failures)
        return 1
    if image_count is None:
        LOGGER.info("Completed successfully")
    else:
        LOGGER.info("Completed successfully (%d image(s))", image_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
