# Optimizing Protocol for Robust Reading Challenges

Existing protocols for Robust Reading Challenges (RRCs) have three limitations:

1. Greedy searches for correspondences that satisfy match criteria can
   limit the performance on complex data sets, such as those with
   overlapping ground truth annotations.
2. Cascaded verification of geometric (IoU) and textual (string match)
   contraints for end-to-end tasks can also limit performance when
   either ground truth annotations or predicted regions overlap.
3. Removing predictions that overlap with ground truth regions marked
   as "don't care" *before* the correspondence search also limits
   performance by irrevocably eliminating them from consideration.
   
This work addresses these issues by unifying the correspondence
matching stage of evaluation in an optimization framework that can
also be tied to the chosen performance metric.

Details may be found in the accompanying paper:

Weinman, J., Gómez Grabowska, A., and Karatzas, D. (2024). Counting
the corner cases: Revisiting robust reading challenge data sets,
evaluation protocols, and metrics. In 18th International Conference
on Document Analysis and Recognition (ICDAR 2024). To appear.

## Installation

After cloning the repository, install the requirements via conda or pip.

### Conda

```shell
conda env create -f conda-environment.yaml
conda activate rrc-evaluation
```

Note that Python 3.9.18 is specified as the version currently running
on the [RRC server](https://rrc.cvc.uab.es) (for consistency).

### Pip

```shell
pip install -r requirements.txt
```

## Data Format

Whereas each RRC (i.e., ICDAR13, ICDAR15, MLT, ArT, LSVT, HierText, OOV) 
has its own format for specifying ground truth, we have simplified
their commonalities into a single universal JSON file format:

```json
{ "IMG_KEY": 
  [ 
    { "points": [[x0,y0], [x1,y1], ... [xN,yN]],
      "text": "Transcription",
      "ignore": false }
    ...
  ],
  ...
}
```

There is a unique key (i.e., `"IMG_KEY"`) for each test image that
refers to a list of word data in that image. Data for each word is
comprised of a mapping containing the bounding polygon coordinates of
the word in the image (`"points"`) and the textual transcription
annotation in Unicode (`"text"`), if applicable. The additional boolean
flag `"ignore"` indicates the marked region as a "don't care" that
will be discounted in evaluation. The `"text"` field is ignored in
that case.

The prediction file format is identical except there is no `"ignore"` field.

Detection-only tasks may also safely omit the `"text"` field from both
ground truth and prediction files.

## Usage

To run the evaluation we need a ground truth file, a prediction file,
and the task (i.e., detection-only or end-to-end). The variety of
additional protocol configuration options can be revealed with the
help flag `--help` or `-h`.

Defaults optimize for the F-score; in the end-to-end it uses
case-sensitive exact string matching and 1-NED for string evaluations.

### Detection Examples

This example uses the organizers' baseline submission (OpenCV 3.0 + Tesseract) for the 2015 End-to-End Focused Scene Text Challenge on the ICDAR13 data set.

The command 
```shell
python ./eval.py --gt data/icdar13_test.json --pred data/icdar13_baseline.json --task det
```
produces the output
```json
{"recall": 0.5736095965103599, "precision": 0.7769571639586411, "fscore": 0.6599749058971143, "tightness": 0.8563325673619869, "quality": 0.5651580055613616, "tp": 526, "total_pred": 677, "total_gt": 917, "total_tightness": 450.43093043240515}
```

To optimize for Panoptic Quality (PQ) `"quality"`, one might instead use the IoU-based correspondence optimization---Equation (12) in the accompanying paper:
```shell
python ./eval.py --gt data/icdar13_test.json --pred data/icdar13_baseline.json --task det --score-fun iou
```

which in this case produces a small increase in the `"quality"` measure.
```json
{"recall": 0.5736095965103599, "precision": 0.7769571639586411, "fscore": 0.6599749058971143, "tightness": 0.8565077882758576, "quality": 0.5652736469675046, "tp": 526, "total_pred": 677, "total_gt": 917, "total_tightness": 450.5230966331011}
```

### Recognition Examples

This example uses the organizers' baseline submission (OpenCV 3.0 + Tesseract) for the 2015 End-to-End Incidental Scene Text Challenge.

The command
```shell
python ./eval.py --gt data/icdar15_test.json --pred data/icdar15_baseline.json --task detrec
```
produces the output
```json
{"recall": 0.03514684641309581, "precision": 0.13419117647058823, "fscore": 0.055703929797787106, "tightness": 0.7674982332585365, "quality": 0.04275266770535915, "char_accuracy": 1.0, "char_quality": 0.04275266770535915, "cned": 0.028649921507064365, "tp": 73, "total_pred": 544, "total_gt": 2077, "total_tightness": 56.027371027873166, "total_rec_score": 73.0}
```

These results are relatively poor because the competition's string
match functions are more lenient than the script defaults. To connect
them, load the example external module `icdar15.py` (usable for both
`icdar13_test.json` and `icdar15_test.json`). 

With this module
```shell
python ./eval.py --gt data/icdar15_test.json --pred data/icdar15_baseline.json --task detrec --external-module icdar15
```
the improved output is produced
```json
{"recall": 0.05055368319691863, "precision": 0.19301470588235295, "fscore": 0.08012209080503624, "tightness": 0.7633869582634346, "quality": 0.06116415918936332, "char_accuracy": 1.0, "char_quality": 0.06116415918936332, "cned": 0.0417329093799682, "tp": 105, "total_pred": 544, "total_gt": 2077, "total_tightness": 80.15563061766063, "total_rec_score": 105.0}
```
Note that `"char_accuracy"` is 1 in both cases because it is measured
over the true positives, which by default are constrained to have
matching strings.  To get a sense of the overall character accuracy,
one could use the 1-NED based correspondence optimization---Equation (11) in the accompanying paper---which optimizes for the overall complementary normalized edit distance (`"cned"`) measure used by ArT and LSVT (distinct from the tightness-aware PCQ measure):

```shell
python ./eval.py --gt data/icdar15_test.json --pred data/icdar15_baseline.json --task detrec --external-module icdar15 --score-fun cned --no-string-match
```
which produces output
```json
{"recall": 0.12084737602311026, "precision": 0.46139705882352944, "fscore": 0.19152995040061047, "tightness": 0.7177377496347613, "quality": 0.1374682755881916, "char_accuracy": 0.7638230213849394, "char_quality": 0.10500143360435002, "cned" 0.0808943368639746, "tp": 251, "total_pred": 544, "total_gt": 2077, "total_tightness": 180.1521751583251, "total_rec_score": 191.7195783676198}
```
It is important to recognize that in this case the recall, precision, and F-score represent *detection* measures, because exact string matching was not required for correspondence. 

To optimize for PCQ we can use the IoU*CNED correspondence optimization---Equation (13) in the accompanying paper:
```shell
python ./eval.py --gt data/icdar15_test.json --pred data/icdar15_baseline.json --task detrec --external-module icdar15 --score-fun iou*cned --no-string-match
```
which happens to yield the same result in this case:
```json
{"recall": 0.12084737602311026, "precision": 0.46139705882352944, "fscore": 0.19152995040061047, "tightness": 0.7177377496347613, "quality": 0.1374682755881916, "char_accuracy": 0.7638230213849394, "char_quality": 0.10500143360435002, "cned": 0.0808943368639746, "tp": 251, "total_pred": 544, "total_gt": 2077, "total_tightness": 180.1521751583251, "total_rec_score": 191.7195783676198}
```

## References

If you make use of this repository in your own research, we would appreciate a citation to the aforementioned accompanying paper:

```bibtex
@inproceedings{ weinman2024counting,
   authors = {Weinman, Jerod and Gómez Grabowska, Amelia and Karatzas, Dimosthenis},
   title = {Counting the corner cases: Revisiting robust reading challenge data sets, evaluation protocols, and metrics},
   booktitle = {18th International Conference on Document Analysis and Recognition ({ICDAR} 2024)},
   series = {Lecture Notes in Computer Science},
   publisher = {Springer},
   location = {Athens, Greece},
   year = {2024}
   note = {To appear}
}
```
