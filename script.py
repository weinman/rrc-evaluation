"""Functions required for RRC Platform
cf https://github.com/sergirobles/rrcevaluation/blob/master/docs/EVALUATIONSCRIPT.md

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


import eval as rrc

from typing import Any
from functools import partial

def default_evaluation_params() -> dict[str,Any]:
    """Default evaluation parameters"""
    return {
        'iou_threshold': 0.5,
        'ignore_overlap_threshold': 0.5,
        'task': 'det', # or 'detrec' for E2E
        }


def validate_data( gtFilePath: str,
                   submFilePath: str,
                   evaluationParams: dict[str,Any]):
    """Determine whether ground truth and/or submission files are correct
    Arguments
      gtFilePath: Path to the ground truth JSON file
      submFilePath: Path to the submitted results JSON file
      evaluationParams: Dict of parameters for the evaluation
    Returns
      Nothing, but raises errors if there is any problem
      """
    return


    

def evaluate_method( gtFilePath: str,
                     submFilePath: str,
                     evaluationParams: dict) -> dict:
    """Evaluate submission and produce metric results
    Arguments
      gtFilePath: Path to the ground truth JSON file
      submFilePath: Path to the submitted results JSON file
      evaluationParams: Dict of parameters for the evaluation
    Returns
      Dict with keys
      'result'(bool) : Indicates the evaluation was completed
      'msg'(str): Description of error (if any)
      'method'(dict[str,float]): results with overall task metrics as keys
      'per_sample'(dict[str,dict[str,float]]) results with per-image metrics
    """
    def default_metrics(task):
        """Get list of default metrics to initialize results in case evaluation fails"""
        metrics = [
            'quality',
            'tightness',
            'fscore',
            'precision',
            'recall',
            'occluded_recall',
            'occluded_fscore']
        return metrics

    task = evaluationParams['task']
    
    gt_anno = rrc.load_ground_truth( gtFilePath, task )
    preds = rrc.load_predictions( submFilePath, task )

    preds = rrc.calculate_prediction_ignores( gt_anno, preds,
                                              evaluationParams['ignore_overlap_threshold'] )

    can_match_fn = partial( rrc.can_match,
                            task=task,
                            iou_threshold=evaluationParams['iou_threshold'] )

    # Use 1.0 to optimize for F-score as the competition metric, or 
    # use iou to optimize for HierText Panoptic Quality (PQ)
    score_match_fn = lambda g,d,iou: 1.0
    
    try:
        overall,per_image = rrc.evaluate( gt_anno, preds,
           task, can_match_fn, score_match=score_match_fn )
    except Exception as e:
        return { 'result': False,
                 'msg': str(e),
                 'method': { m:0.0 for m in default_metrics(task)},
                 'per_sample': { k: { m:0.0 for m in default_metrics(task)} 
                                 for k in gt_anno}
        }
    return {'result': True,
            'msg': 'Completed',
            'method': overall, 
            'per_sample': per_image
           }
