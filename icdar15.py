

# Functions default_evaluation_params and transcription_match come
# from official evaluation file script.py hosted on the rrc server:
# https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=4
# and are subject to the original license terms of that work.

# Function transcription_score is a derivative work of function
# transcription_match.


from typing import Any

from pyeditdistance.distance import normalized_levenshtein  # type: ignore


def default_evaluation_params() -> dict[str,Any]:
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """          
    return {
            'IOU_CONSTRAINT' :0.5,
            'AREA_PRECISION_CONSTRAINT' :0.5,
            'WORD_SPOTTING' :False,
            'MIN_LENGTH_CARE_WORD' :3,
            'GT_SAMPLE_NAME_2_ID':'gt_img_([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID':'res_img_([0-9]+).txt',            
            'LTRB':False, #LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'CRLF':False, # Lines are delimited by Windows CRLF format
            'CONFIDENCES':False, #Detections must include confidence value. AP will be calculated,
            'SPECIAL_CHARACTERS':'!?.:,*"()·[]/\'',
            'ONLY_REMOVE_FIRST_LAST_CHARACTER' : True
        }


# Note optional arguments have the same values as the default_evaluation_params
# used above
def transcription_match(transGt: str, transDet: str,
                        specialCharacters: str='!?.:,*"()·[]/\'',
                        onlyRemoveFirstLastCharacterGT: bool=True) -> bool:

    transGt = transGt.upper()
    transDet = transDet.upper()

    if onlyRemoveFirstLastCharacterGT:
        #special characters in GT are allowed only at initial or final position
        if (transGt==transDet):
            return True        

        if specialCharacters.find(transGt[0])>-1:
            if transGt[1:]==transDet:
                return True

        if specialCharacters.find(transGt[-1])>-1:
            if transGt[0:len(transGt)-1]==transDet:
                return True

        if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1:
            if transGt[1:len(transGt)-1]==transDet:
                return True
        return False
    else:
        #Special characters are removed from the begining and the end of both Detection and GroundTruth
        while len(transGt)>0 and specialCharacters.find(transGt[0])>-1:
            transGt = transGt[1:]
				
        while len(transDet)>0 and specialCharacters.find(transDet[0])>-1:
            transDet = transDet[1:]
                
        while len(transGt)>0 and specialCharacters.find(transGt[-1])>-1 :
            transGt = transGt[0:len(transGt)-1]
                
        while len(transDet)>0 and specialCharacters.find(transDet[-1])>-1:
            transDet = transDet[0:len(transDet)-1]
                
        return transGt == transDet


def transcription_score(transGt: str, transDet: str,
                        specialCharacters: str='!?.:,*"()·[]/\'',
                        onlyRemoveFirstLastCharacterGT: bool=True) -> float:
    """Calculate 1-NED with appropriate penalty-free transformations"""
    transGt = transGt.upper()
    transDet = transDet.upper()

    def str_score(p,d):
        return 1 - normalized_levenshtein(p,d)

    if transcription_match(transGt, transDet):
        return 1.0
    
    if onlyRemoveFirstLastCharacterGT:
        #special characters in GT are allowed only at initial or final position

        # check front and back char
        if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1:
            return str_score( transGt[1:len(transGt)-1], transDet)
        # not both, check back only
        elif  specialCharacters.find(transGt[-1])>-1:
            return str_score(transGt[0:len(transGt)-1], transDet)
        # not back, check front
        elif specialCharacters.find(transGt[0])>-1:
            return str_score( transGt[1:], transDet)
        else: # no special characters
            return str_score( transGt, transDet )
    else:
        #Special characters are removed from the begining and the end of both Detection and GroundTruth
        while len(transGt)>0 and specialCharacters.find(transGt[0])>-1:
            transGt = transGt[1:]
				
        while len(transDet)>0 and specialCharacters.find(transDet[0])>-1:
            transDet = transDet[1:]
                
        while len(transGt)>0 and specialCharacters.find(transGt[-1])>-1 :
            transGt = transGt[0:len(transGt)-1]
                
        while len(transDet)>0 and specialCharacters.find(transDet[-1])>-1:
            transDet = transDet[0:len(transDet)-1]
                
        return str_score(transGt,transDet)
