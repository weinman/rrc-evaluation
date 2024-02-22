"""Command-line script for standalone ICDAR competition evaluation

Copyright 2024 Jerod Weinman

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>."""

import json

from typing import Union, Tuple, Any, Optional, Callable

import re
import logging
import importlib
import argparse

import scipy  # type: ignore
import numpy as np
import numpy.typing as npt
from Polygon import Polygon  # type: ignore

from pyeditdistance.distance import normalized_levenshtein  # type: ignore


# Minimum area of a polygon to be considered for area-based processing
# Values below this will be treated as an IoU or overlap percentage of zero.
MIN_POLY_AREA = 1e-4

parser = argparse.ArgumentParser(
    description='Robust Reading Competition Task Evaluation')
parser.add_argument('--gt', type=str, required=True,
                    help="Path to the ground truth JSON file")
parser.add_argument('--pred', type=str, required=True,
                    help="Path to the predictions JSON file")
parser.add_argument('--task', type=str, required=True,
                    choices=['det', 'detrec'],
                    help="Task to evaluate against")
parser.add_argument('--output', type=str, required=False, default=None,
                    help="Path to the JSON file containing results")
parser.add_argument('--score-fun', type=str, required=False, default='one',
                    choices=['one','iou','iou*cned','cned'],
                    help="Constraint-satisfying correspondence score")
parser.add_argument('--string-match', default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Require strings to match in detrec (end-to-end) task")
parser.add_argument('--iou-threshold', type=float, default=0.5,
                    help="Minimum IoU for elements to be considered a match")
parser.add_argument('--overlap-threshold', type=float, default=0.5,
                    help="Minimum overlap with a \"don't care\" GT to ignore detection")
parser.add_argument('--use-case', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='Evaluate recognition in a case-sensitive manner')
parser.add_argument('--external-module', type=str, required=False, default=None,
                    help='Use external transcription match/score functions')
parser.add_argument('--cast-points-int', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='Cast prediction coordinates to int type for processing')
parser.add_argument('--gt-regex', type=str, required=False,
                    help="Regular expression to filter image keys for evaluation")


# Type aliases for hints
# NB: "type" omitted for compatibility with Python3.9, used by RRC platform
WordData = dict[str, Any]
ImageData = list[WordData]
Number = Union[int, float]


def verify_args( args: argparse.Namespace ):
    """Verify command-line arguments are self-consistent and have valid values"""

    #if args.string_match and 'rec' not in args.task:
        # This is not a fatal error, but it doesn't make sense and could
        # signal a user error/oversight
        #logging.warning("string-match set for non-recognition task")

    if args.external_module and 'rec' not in args.task:
        # This is not a fatal error, but it doesn't make sense and could
        # signal a user error/oversight
        logging.warning("external-module set to load string functions for" +
                        "non-recognition task")

    if 'cned' in args.score_fun and 'rec' not in args.task:
        raise ValueError("CNED scores can only be used with detrec (E2E) task ")

    if args.iou_threshold < 0 or args.iou_threshold > 1:
        raise ValueError("IoU threshold must be in [0,1]")

    if args.overlap_threshold < 0 or args.overlap_threshold > 1:
        raise ValueError("Overlap threshold must be in [0,1]")


def warn_image_keys( gt_keys: set,
                     preds_keys: set ):
    """Log warnings about image key discrepancies between ground truth and
    prediction files (i.e., predictions missing an image from ground truth)."""

    for gt in gt_keys:
        if gt not in preds_keys:
            logging.warning("Key %s missing from predictions", gt)
    for pred in preds_keys:
        if pred not in gt_keys:
            logging.warning(f"Extra image '{pred}' in predictions")


def add_word_polygons( data: dict[str,ImageData],
                       int_cast: bool = False):
    """Add the field 'geometry', a polygon.Polygon, to each word.
    It is constructed from the 'points' of the word.
    Arguments:
      data: The dict of Image data
      int_cast: Whether to cast the point values to ints

    Returns
      Nothing; called for side-effect (mutating data)
    """
    for words in data.values():
        for word in words:
            points = word['points']
            if int_cast:
                points = [ [int(p[0]), int(p[1])] for p in points]
            word['geometry'] = Polygon(points)


def load_ground_truth( gt_file: str,
                       is_e2e: bool,
                       image_regex: Optional[str] = None,
                       use_case: Optional[bool] = True,
                       points_int_cast: bool = False) \
                       -> dict[str,ImageData]:
    """Load the ground truth file

    Arguments
      gt_file : Path to the ground truth JSON file
      is_e2e : Whether the evaluation is end-to-end (i.e., includes recognition)
      image_regex : Regular expression to filter image keys (default=None)
      gt_anno : Dict indexed by the image id, giving the list of words
    """

    logging.info("Loading ground truth annotations...")

    with open(gt_file, encoding='utf-8') as fd:
        gt = json.load(fd)

    if image_regex:  # Filter to matching image keys
        regex = re.compile(image_regex)
        gt = { k:v for (k,v) in gt.items() if regex.match(k) }

    if len(gt) == 0:
        raise ValueError("No ground truth images for evaluation")

    # Add additional/transformed fields for evaluation processing
    add_word_polygons(gt,points_int_cast)

    if is_e2e and not use_case:  # Convert all to upper when ignoring case
        for img in gt:
            for word in gt[img]:
                word['text'] = word['text'].upper()
    return gt


def load_predictions( preds_file: str,
                      is_e2e: bool,
                      image_regex: Optional[str] = None,
                      use_case: Optional[bool] = True,
                      points_int_cast: bool = False) -> dict[str,ImageData]:
    """Load the predictions file

    Arguments
      preds_file : Path to the predictions JSON file
      is_e2e:      Whether the evaluation is end-to-end (i.e., includes
                      recognition)
      image_regex: Regular expression string to filter in matches
      use_case:    Whether to preserve case in transcription
    Returns
      preds : Dict indexed by the image id, giving the list of words
    """

    logging.info("Loading predictions...")

    with open(preds_file, encoding='utf-8') as fd:
        preds = json.load(fd)

    if image_regex:  # Filter to matching image keys
        regex = re.compile(image_regex)
        preds = { k:v for (k,v) in preds.items() if regex.match(k) }

    # Add additional/transformed fields for evaluation processing
    add_word_polygons(preds,points_int_cast)

    if is_e2e and not use_case:  # Convert all to upper when ignoring case
        for img in preds:
            for word in preds[img]:
                word['text'] = word['text'].upper()

    return preds


def calculate_prediction_ignores( gts: dict[str,ImageData],
                                  preds: dict[str,ImageData],
                                  area_thresh: float) -> dict[str,ImageData]:
    """Set all prediction ignore flags using the area precision constraint:
         p.ignore == Ǝ g such that g.ignore and |p∩g|/|p| > area_thresh

    Arguments
      gt :  List of dicts containing ground truth elements (each has the field
           'geometry' and 'ignore' among others).
      pred : List of dicts containing predicted elements (each has the field
             'geometry' among others).
      area_thresh : Threshold the calculated precision must exceed for ignore
                     to be set true
    Returns
      pred : Input with each word's ignore field set according to criteria above
    """
    def calc_overlap( p: Polygon, g: Polygon ) -> float :
        """ Return the overlap of p on g """
        intersection = p & g
        if len(intersection)==0 or p.area() < MIN_POLY_AREA:
            return 0.0
        else:
            return intersection.area() / p.area()

    for img in preds:
        if img not in gts:
            continue  # Warning about excess image key already issued
        gt = gts[img]
        for p in preds[img]:
            p['ignore'] = False  # Set default value of field
            for g in gt:
                if g['ignore'] and \
                   calc_overlap(p['geometry'], g['geometry']) > area_thresh:
                    p['ignore'] = True
                    break
    return preds


def calc_iou( p: Polygon, q: Polygon ) -> float :
    """ Return the IoU between two shapes """
    r = p & q
    if len(r)==0 or q.area() < MIN_POLY_AREA or p.area() < MIN_POLY_AREA:
        return 0.0
    else:
        intersection = r.area()
        union = p.area() + q.area() - r.area()
        return intersection / union


def calc_score_pairs( gt: list[WordData],
                      pred: list[WordData],
                      can_match: Callable[[WordData,WordData,float],bool],
                      score_match: Callable[[WordData,WordData,float],float] ) \
                      -> Tuple[npt.NDArray[np.bool_],
                               npt.NDArray[np.double],
                               npt.NDArray[np.double]]:
    """Return the correspondence score between all pairs of shapes.

    Arguments
      gt :  List of dicts containing ground truth elements (each has the field
           'geometry' among others).
      pred : List of dicts containing predicted elements (each has the field
             'geometry' among others).
    can_match
      score_iou: Whether to factor IoU into correspondence scores
      str_score: For E2E, function for string portion of score
    Returns
      allowed: MxN numpy bool array of can_match(g,d) correspondence candidates
      scores : MxN numpy float array of compatibility scores
      ious : MxN numpy float array of IoU values

      where M is len(gt) and N is len(pred).
    """
    allowed = np.zeros( (len(gt),len(pred)), dtype=np.bool_ )
    scores = -np.ones( (len(gt),len(pred)), dtype=np.double )
    ious = np.zeros( (len(gt),len(pred)), dtype=np.double )

    for i,gt_el in enumerate(gt):
        for j,pred_el in enumerate(pred):
            try:
                the_iou = calc_iou( gt_el['geometry'], pred_el['geometry'])
            except Exception as e:
                logging.warning('Error at iou(%d,%d): %s}. Skipping ...',i,j,e)
                continue

            if the_iou != 0:
                ious[i,j] = the_iou

            allowed[i,j] = can_match( gt_el, pred_el, the_iou)

            if allowed[i,j]:
                scores[i,j] = score_match( gt_el, pred_el, the_iou)

    return allowed,scores,ious


def get_stats( num_tp: Number, num_gt: Number, num_pred: Number,
               tot_iou: Number, prefix: str = '') -> dict[str,float]:
    """Calculate statistics: recall, precision, fscore, tightness, and quality
    from accumulated totals.

    Arguments
      num_tp   : Number of true positives
      num_gt   : Number of ground truth positives in the evaluation
      num_pred : Number of predicted positives in the evaluation
      tot_iou  : Total IoU scores among true positives
      prefix   : Optional prefix for return result keys (default='')
    Returns
      dict containing statistics with keys 'recall', 'precision', 'fscore',
        'tightness' (average IoU score), and 'quality' (product of fscore and
        tightness).
    """
    recall    = float(num_tp) / num_gt   if (num_gt > 0)   else 0.0
    precision = float(num_tp) / num_pred if (num_pred > 0) else 0.0
    tightness = tot_iou / float(num_tp)  if (num_tp > 0)   else 0.0
    fscore    = 2.0*recall*precision / (recall+precision) \
        if (recall + precision > 0) else 0.0
    quality = tightness * fscore

    stats = {prefix+'recall'    : recall,
             prefix+'precision' : precision,
             prefix+'fscore'    : fscore,
             prefix+'tightness' : tightness,
             prefix+'quality'   : quality }
    return stats


def get_final_stats(totals: dict[str,Number],
                    task : str) -> dict[str,Number] :
    """Process totals to produce final statistics for the entire data set.

    Arguments
      totals : Dict with keys 'tp', 'total_gt', 'total_pred',
                 'total_tightness', and (if 'rec' in task), 'total_rec_score'.
      task : String containing a valid task (cf argparser)
    Returns
      dict containing statistics with keys 'recall', 'precision',
        'fscore', 'tightness' (average IoU score),  'quality'
        (product of fscore and tightness), and (if 'rec' in task)
       'char_accuracy' and 'char_quality' (product of quality and
       char_accuracy).
    """
    final_stats = get_stats( totals['tp'],
                             totals['total_gt'],
                             totals['total_pred'],
                             totals['total_tightness'])
    if 'rec' in task:
        if totals['tp'] > 0:
            accuracy = totals['total_rec_score'] / float(totals['tp'])
        else:
            accuracy = 0.0
        final_stats['char_accuracy'] = accuracy
        final_stats['char_quality'] = accuracy * final_stats['quality']
        final_stats['cned'] = totals['total_rec_score'] / \
            (totals['total_gt'] + totals['total_pred'] - totals['tp'])
        # cned denom = |TP| + |FN| + |FP|
        #        |G| = |TP| + |FN|,
        #        |D| = |TP| + |FP|,  |FP| = |D| - |TP|
    return final_stats


def find_matches(allowable: npt.NDArray[np.bool_],
                 scores: npt.NDArray[np.double],
                 ious: npt.NDArray[np.double] ) \
                 -> Tuple[npt.NDArray[np.uint],
                          npt.NDArray[np.uint],
                          npt.NDArray[np.double]]:
    """Optimize the bipartite matches and filter them to allowable matches.
    Parameters
      allowable:      MxN numpy bool array of valid correspondence candidates
      scores:         MxN numpy float array of match candidate scores
      ious:           MxN numpy float array of IoU scores
    Returns
      matches_gt:   Length T numpy array of values in [0,M) indicating ground
                      truth element matched (corresponds to entries in
                      matches_pred)
      matches_pred: Length T numpy array of values in [0,N) indicating
                      predicted element matched (corresponds to entries in
                      matches_gt)
      matches_ious: Length T numpy array of matches' values from ious
    """
    matches_gt,matches_pred = \
        scipy.optimize.linear_sum_assignment(scores, maximize=True)

    # A maximal bipartite matching, which scipy linear sum assignment algorithm
    # appears to give, may include non-allowable matchings due to lack of
    # alternatives. Therefore, these must be removed from the final list.
    # (This is likely more straightforward than fiddling with returned indices
    # after pre-filtering rows/columns that have no viable partners).
    matches_valid = allowable[matches_gt,matches_pred]
    matches_gt    = matches_gt[matches_valid]
    matches_pred  = matches_pred[matches_valid]

    matches_ious  = ious[matches_gt,matches_pred]

    return matches_gt, matches_pred, matches_ious


def evaluate_image( gt: list[WordData],
                    pred: list[WordData],
                    task: str,
                    can_match: Callable[[WordData,WordData,float],bool],
                    score_match: Callable[[WordData,WordData,float],float],
                    str_score: Optional[Callable[[str,str],float]] = None) \
                    -> Tuple[dict[str,Number], dict[str,Number]]:
    """Apply the appropriate evaluation scheme to lists of ground truth and
    prediction elements from the same image.

    Arguments
      gt : List of dicts containing ground truth elements (each has the fields
           'geometry', 'text', and 'ignore').
      pred : List of dicts containing predicted elements for evaluation (each
             has the fields 'geometry' and (if task contains 'rec') 'text'.
      task : string describing the task (det, detrec)
    Returns
      results : dict containing totals for the accumulator
      stats : dict containing statistics for this image
    """
    allowed, scores, ious = calc_score_pairs( gt, pred, can_match, score_match )
    matches_gt, matches_pred, matches_ious = find_matches(allowed, scores, ious)

    # Count the total number of ground truth entries marked as ignore
    num_gt_ignore = len( [ el for el in gt if el['ignore'] ] )

    # Count the number of unmatched predictions marked as ignorable
    preds_ignore = np.asarray( [el['ignore'] for el in pred], dtype=bool)
    preds_unmatched = np.ones( len(pred), dtype=bool )
    preds_unmatched[matches_pred] = False
    num_unmatched_preds_ignore = int(np.sum( np.logical_and( preds_unmatched,
                                                             preds_ignore )))

    # Discount GT ignores, but preserve successful matches where prediction
    # happened to overlap with a different GT marked ignore
    total_pred = len(pred) - num_unmatched_preds_ignore
    total_gt   = len(gt)   - num_gt_ignore

    num_tp = len(matches_pred)

    # Accumulate tightness for matches that count (not ignorable)
    total_tightness = float(np.sum(matches_ious))

    results = { 'tp' : int(num_tp),
                'total_gt' : int(total_gt),
                'total_pred' :  int(total_pred),
                'total_tightness' : total_tightness }

    stats = get_stats( num_tp, total_gt, total_pred, total_tightness )

    if 'rec' in task:
        assert str_score is not None, \
            "str_score must be defined for task 'detrec'"
        # measure text (mis)prediction (i.e., 1-CER) for true positives
        text_score_matches = [ str_score( gt[g]['text'], pred[p]['text'] )
          for (g,p) in zip(matches_gt,matches_pred) ]
        # tally scores among true positives
        total_rec_score = sum( text_score_matches )

        accuracy = total_rec_score / float(num_tp) if (num_tp > 0) else 0.0

        stats['char_accuracy'] = accuracy
        stats['char_quality']  = accuracy * stats['quality']

        results['total_rec_score'] = total_rec_score

    return results, stats


def evaluate(gt: dict[str,ImageData],
             pred: dict[str,ImageData],
             task: str,
             can_match: Callable[[WordData,WordData,float],bool],
             score_match: Callable[[WordData,WordData,float],float],
             str_score: Optional[Callable[[str,str],float]] = None ) \
             -> Tuple[dict[str,float], dict[str,dict[str,float]]]:
    """Run the primary evaluation protocol over all images.

    Returns:
      final_stats : dict containing pooled statistics for the entire data set
      stats : dict containing statistics for each image in the data set
    """

    def accumulate( totals: dict[str,float], results: dict[str,float] ):
        """Side-effect totals by accumulating matching keys of results"""
        for (k,v) in results.items():
            totals[k] += v

    # initialize accumulator
    totals = { 'tp': 0,
               'total_pred': 0,
               'total_gt': 0,
               'total_tightness': 0.0 }
    if 'rec' in task:
        totals['total_rec_score'] = 0.0

    stats = {}  # Collected per-image statistics

    for (img,gt_words) in gt.items():  # Process each image
        pred_words = pred[img] if img in pred else []

        img_results, img_stats = evaluate_image( gt_words, pred_words,
                                                 task,
                                                 can_match, score_match,
                                                 str_score )
        accumulate( totals, img_results)
        stats[img] = img_stats

    final_stats = get_final_stats( totals, task )  # Process totals
    final_stats.update(totals)  # Include raw numbers

    return final_stats, stats


def process_args(args: argparse.Namespace) -> \
    Tuple[Callable[[WordData,WordData,float],bool],
          Callable[[WordData,WordData,float],float],
          Callable[[str,str],float]]:
    """Process command-line arguments for specific functionality: string
    match (bool) and score (i.e., 1-NED) functions as well as
    correspondence candidate criteria (bool) and match score (float)
    functions.

    Returned function str_score may be the default or loaded from an
    external module. Returned functions can_match and score_match are
    established in part from command-line arguments, though when
    can_match requires strings to match, the str_score function is
    used and when score_match incorporates the 'cned' option, the
    str_score function is used.

    Parameters
      args: The argparse Namespace object resulting from command-line invocation
    Returns
      can_match:    Predicate taking ground truth and predicted word dicts with 
                      their pre-calculated iou score and returning whether the 
                      correspondence satisfies match criteria
      score_match:  Function taking ground truth and predicted word dicts with 
                      their pre-calculated iou score and returning their match 
                      score (assumes they are valid matches)
      str_score:    Function taking two strings to give compatibility score 
                      (i.e., 1-NED)
    """
    # Transcription/String Matching + Distance Functions
    if args.external_module is not None:
        try:
            rrc = importlib.import_module(args.external_module)
        except Exception as e:
            raise ImportError(f'Could not load external module specified {args.external_module}') from e

        is_str_match = rrc.transcription_match  # type: ignore
        str_score = rrc.transcription_score  # type: ignore
    else:
        # Simple equality and complementary normalized edit distance (CNED)
        is_str_match = lambda gs,ds: gs==ds
        str_score = lambda gs,ds: 1 - normalized_levenshtein(gs,ds)

    # TODO: Allow can_match and score_match to be loaded from module
    # for greater extensibility

    # Set up matching functions
    if 'rec' in args.task and args.string_match:
        can_match = lambda g,d,iou: iou > args.iou_threshold and \
            (not g['ignore']) and is_str_match(g['text'], d['text'])
    else:
        can_match = lambda g,d,iou: iou > args.iou_threshold and not g['ignore']

    if args.score_fun=='one':
        score_match = lambda g,d,iou: 1.0
    elif args.score_fun=='iou':
        score_match = lambda g,d,iou: iou
    elif args.score_fun=='cned': # strictly speaking, this name is not precise
        score_match = lambda g,d,iou: str_score(g['text'],d['text'])
    elif args.score_fun=='iou*cned':
        score_match = lambda g,d,iou: iou * str_score(g['text'],d['text'])
    else:
        raise ValueError(f'Unknown score method for argument score-fun: "{args.score_fun}"')  # shouldn't happen due to "choices" in argparser option

    return can_match, score_match, str_score


def main():
    """Main entry point for evaluation script"""

    args = parser.parse_args()

    verify_args(args)

    is_e2e = 'rec' in args.task

    gt_anno = load_ground_truth( args.gt, is_e2e=is_e2e,
                                 image_regex=args.gt_regex,
                                 use_case=args.use_case,
                                 points_int_cast=args.cast_points_int)
    preds = load_predictions( args.pred, is_e2e=is_e2e,
                              image_regex=args.gt_regex,
                              use_case=args.use_case,
                              points_int_cast=args.cast_points_int)
    preds = calculate_prediction_ignores(gt_anno, preds, args.overlap_threshold)

    # Verify we have the same images (key sets)
    if gt_anno.keys() != preds.keys() :
        warn_image_keys( gt_anno.keys(), preds.keys() )

    # Get coordinating functions
    can_match, score_match, str_score = process_args(args)

    overall,per_image = evaluate( gt_anno, preds,
                                  args.task,
                                  can_match,
                                  score_match,
                                  str_score )

    print(overall)

    if args.output:
        with open(args.output,'w',encoding='utf-8') as fd:
            json.dump( {'images': per_image,
                        'results': overall }, fd, indent=4 )


if __name__ == "__main__":
    main()
