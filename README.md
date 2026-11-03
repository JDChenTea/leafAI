# leafAI

Current release: **v1.1**

leafAI is a cross-platform command-line toolkit for quantitative leaf and tea-bud
image analysis. It detects foreground specimens, extracts shape, texture, and color
descriptors, estimates trichome coverage from HSV thresholds, and provides supporting
CSV-based statistical workflows.

## Requirements

- Python 3.9 or later
- Windows, macOS, or Linux
- Input images with a dark, visually separable background are recommended

## Installation

Create an isolated environment in the project directory:

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bat
:: Windows Command Prompt
.venv\Scripts\activate.bat
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python leafAI.py --help
```

The examples below assume that the virtual environment is active. Without activation,
replace `python` with `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe`
on Windows.

## Command structure

```text
python leafAI.py -p MODE [input options] [output options] [mode options]
```

Use one of the following image inputs:

```bash
# One image
-i path/to/image.png

# Every supported image in a directory
-d path/to/images

# Directory input restricted by a glob pattern
-d path/to/images --pattern "sample_R*.png"
```

Supported extensions are PNG, JPEG, BMP, and TIFF. `--out` is an output prefix, not
necessarily a directory. Parent directories are created automatically.

## Recommended image workflow

For an image containing multiple specimens, first inspect detection with `mark`, then
extract individual specimens with `split`, and finally run feature or trichome analysis
on the resulting crops:

```text
source image -> mark -> split -> features / hsv-plot / trichome analysis
```

### Detect and annotate objects: `mark`

```bash
python leafAI.py -p mark \
  -i data/SZA016.png \
  -o result/SZA016 \
  --min-area 3000 \
  --max-area 700000
```

Output:

```text
result/SZA016_marker.png
```

The accepted contours are enclosed by numbered rectangles. Use `--min-area` and
`--max-area` to exclude small debris or large reference objects.

### Deskew and crop objects: `split`

```bash
python leafAI.py -p split \
  -i data/SZA016.png \
  -o result/SZA016 \
  --min-area 3000 \
  --max-area 700000 \
  --border 500 \
  --margin 50
```

Outputs are numbered sequentially:

```text
result/SZA016_R1.png
result/SZA016_R2.png
...
```

The principal axis of each detected object is aligned vertically before its rotated
contour is cropped. Re-running `split` with the same output prefix removes older
`_R<number>.png` crops for that prefix before writing the new set.

Directory mode uses the same operation:

```bash
python leafAI.py -p split \
  -d data \
  --pattern "*.png" \
  -o result/split
```

## Feature extraction

Feature modes analyze the largest foreground contour in each input image. Use cropped
single-specimen images whenever possible.

### Shape descriptors: `shape`

```bash
python leafAI.py -p shape \
  -i result/SZA016_R1.png \
  -o result/features/SZA016_R1 \
  --dpi 600
```

Outputs:

```text
result/features/SZA016_R1.shape.csv
result/features/SZA016_R1_gray.png
result/features/SZA016_R1_binary.png
```

The CSV contains nine descriptors:

| Field | Definition |
|---|---|
| `perimeter` | Contour perimeter in centimetres |
| `area` | Contour area in square centimetres |
| `length` | Long side of the bounding rectangle in centimetres |
| `width` | Short side of the bounding rectangle in centimetres |
| `aspect_ratio` | Length divided by width |
| `extent` | Contour area divided by bounding-rectangle area |
| `solidity` | Contour area divided by convex-hull area |
| `equivalent_diameter` | Diameter of a circle with the same area, in centimetres |
| `circularity` | `4*pi*area/perimeter^2`; 1 represents an ideal circle |

The default is 300 DPI. Physical measurements are valid only when the source DPI is
known and meaningful. For the supplied 600-DPI scans, pass `--dpi 600` explicitly.

### Texture descriptors: `texture`

```bash
python leafAI.py -p texture \
  -i result/SZA016_R1.png \
  -o result/features/SZA016_R1 \
  --texture-levels 32
```

The grayscale image is quantized to 32 levels by default. Valid values are 8, 16, 32,
64, 128, and 256. Co-occurrence information from 0, 45, 90, and 135 degrees is
aggregated into a symmetric GLCM. The output contains contrast, dissimilarity,
homogeneity, correlation, angular second moment, energy, and entropy.

### Color descriptors: `color`

```bash
python leafAI.py -p color \
  -i result/SZA016_R1.png \
  -o result/features/SZA016_R1
```

For each channel in HSV, RGB, and CIE Lab, leafAI reports the mean, population standard
deviation, and cube root of the third central moment. This produces 27 descriptors:

- HSV: H in degrees and S/V as percentages
- RGB: channel values on the 0-255 scale
- CIE Lab: L* on the 0-100 scale and signed a*/b* values

Color and texture measurements use a lightly eroded foreground mask to reduce
segmentation-edge contamination.

### Combined descriptors: `features`

```bash
python leafAI.py -p features \
  -d result \
  --pattern "SZA016_R*.png" \
  -o result/features/SZA016 \
  --dpi 600 \
  --texture-levels 32
```

The combined CSV contains 43 descriptors per image: 9 shape, 7 texture, and 27 color
features.

## Trichome analysis

### Quantify trichome coverage: `trichome-index`

```bash
python leafAI.py -p trichome-index \
  -i result/SZA016_R1.png \
  --hsv-thresh 0,160,0,60,50,255 \
  -o result/index/SZA016_R1
```

The six values are ordered as:

```text
Hmin,Hmax,Smin,Smax,Vmin,Vmax
```

The default interval is `0,160,0,60,50,255`. The output CSV contains the matching
pixel count, total foreground pixel count, and percentage trichome index.

Batch example:

```bash
python leafAI.py -p trichome-index \
  -d result \
  --pattern "SZA016_R*.png" \
  --hsv-thresh 0,160,0,60,50,255 \
  -o result/index/SZA016
```

## Tabular analysis

### Aggregate replicate measurements: `summarize`

```bash
python leafAI.py -p summarize \
  -v data/GX2022.trichome.csv \
  -g data/GX2022.group.csv \
  -o result/GX2022.summary.csv
```

The measurement CSV must have an ID column followed by one or more numeric traits:

```csv
ID,trichome_pixel,bud_pixel,trichome_index
sample_1,1200,2000,60.0
sample_2,1300,2100,61.9
```

The group CSV must contain sample ID and group ID in its first two columns. A header is
optional:

```csv
ID,group
sample_1,line_A
sample_2,line_A
```

For every trait and group, leafAI reports the mean, minimum, maximum, sample standard
deviation, and replicate count.

### Describe and plot a trait: `distribution`

```bash
python leafAI.py -p distribution \
  -v result/GX2022.summary.csv \
  -t trichome_index_mean \
  -o result/GX2022.trichome_index_mean.png
```

The terminal reports count, minimum, maximum, mean, sample standard deviation, and
coefficient of variation. The output figure contains a histogram and kernel-density
estimate.

### Join measurements by sample ID: `join`

When both files use the same trait name:

```bash
python leafAI.py -p join \
  -a repeat1.csv \
  -b repeat2.csv \
  -t trichome_index \
  -c repeat1,repeat2 \
  -o result/repeats.csv
```

When the trait names differ:

```bash
python leafAI.py -p join \
  -a first.csv \
  -b second.csv \
  --left-trait trichome_index_mean \
  --right-trait trichome_index \
  -c first_measurement,second_measurement \
  -o result/repeats.csv
```

Only IDs present in both files are written. `--columns`/`-c` supplies the two output
trait names and is recommended whenever the source trait names are identical.

### Assess repeatability: `correlation`

```bash
python leafAI.py -p correlation \
  -v result/repeats.csv \
  -o result/repeats.correlation.png
```

The input must contain ID in the first column and the two numeric measurements in the
second and third columns. leafAI reports Pearson's correlation coefficient and its
p-value, then produces a scatter plot with linear regression and confidence interval.

