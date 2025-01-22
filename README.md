

## Dependencies

leafAI requires Python3.8 and the following modules.

```bash
pip install opencv-python
pip install scikit-learn
pip install scikit-image
pip install pandas
pip install seaborn
```

## Usage

Use ImageRecognitionThreshold.py to annotate the bud or leaf.

```bash
python scripts/ImageRecognitionThreshold.py -i input.png

# for example
python scripts/ImageRecognitionThreshold.py -i test/MG002.png
# output: MG002_marker.png
```

Use ImageSplit.py to segment the image.

~~~bash
python scripts/ImageSplit.py -i input.png

# for example
python scripts/ImageSplit.py -i test/MG002.png
# output: MG002_R*.png
~~~

Use leafAI.py to extract the morphological features of the image.

~~~bash
python leafAI.py -p shape [-i input_R*.png | -d inputdir] -o output

# for example
python leafAI.py -p shape -i test/MG002_R1.png -o MG002_R1
# output: MG002_R1.shape.csv
ID,perimeter,area,length,width,aspectRatio,extent,solidity,equivalentDiameter,roundness
MG002_R1,5.28,1.51,1.26,1.28,1.02,0.93,0.96,1.39,1.47

python leafAI.py -p shape -d test -o MG002 
# output: MG002.shape.csv
ID,perimeter,area,length,width,aspectRatio,extent,solidity,equivalentDiameter,roundness
MG002_R19,7.04,0.84,3.26,0.39,0.12,0.66,0.91,1.03,4.69
MG002_R21,8.12,0.9,3.73,0.44,0.12,0.55,0.77,1.07,5.84
MG002_R16,7.32,0.79,3.38,0.36,0.11,0.65,0.85,1.01,5.37
MG002_R27,7.14,0.71,3.25,0.42,0.13,0.51,0.73,0.95,5.75
......
~~~

