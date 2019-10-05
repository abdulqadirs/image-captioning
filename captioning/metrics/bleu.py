from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def bleu_score(reference_caption, predicted_caption):
    """
    calculates the bleu score using nltk
    (calculates for one caption at a time)

    Params
    ------
    - reference_caption: ground trugh
    - predicted_caption: the caption predicted by the model

    Return
    ------
    - bleu score 
    """

    reference_caption = reference_caption.squeeze(0)
    reference_caption = [caption.item() for caption in reference_caption]
    sf = SmoothingFunction()
    bleu = sentence_bleu([reference_caption], predicted_caption, smoothing_function=sf.method2)

    return bleu