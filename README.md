# Evaluation Protocol for Occluded RoadText Robust Reading Challenge

This is the official evaluation protocol for the ICDAR'24 [Occluded RoadText
Robust Reading Challenge](https://rrc.cvc.uab.es/?ch=29).  For additional
details about the tasks, data, and formats, please refer to the challenge web
page.

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

## Usage

To run the evaluation we need a ground truth file, a prediction file, and the
task (i.e., detection-only Task 1 `det` or end-to-end Tasks 2 and 3
`detrec`). The variety of additional protocol configuration options can be
revealed with the help flag `--help` or `-h`.

```shell
python ./eval.py --gt PATH_TO_GT --pred PATH_TO_RESULTS --task detrec
```

## References

If you make use of this repository in your own research, we would appreciate a
citation to the accompanying competition report:

```bibtex
@inproceedings{tom2024occluded,
   authors = {Tom, George and Mathew, Minesh and
   Mondal, Ajoy and Karatzas, Dimosthenes and Jawahar, C. V. and Weinman, Jerod}
   title = {ICDAR2024 Challenge on Occluded RoadText}
   note = {Forthcoming}
   year = {2024}
}
```

For background on the general form of evaluation, please consult the [upstream
repository](https://github.com/weinman/rrc-evaluation) and/or its accompanying paper:

```bibtex
@inproceedings{ weinman2024counting,
   authors = {Weinman, Jerod and {Gómez Grabowska}, Amelia and Karatzas, Dimosthenes}
   title = {Counting the corner cases: Revisiting robust reading challenge data sets, evaluation protocols, and metrics}
   note = {Under review}
   year = {2024}
}
```
