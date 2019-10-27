from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def bleu_score(reference_caption, predicted_caption):
    """
    Calculates the bleu score using nltk (one caption at a time)

    Args:
        reference_caption (tensor): ground truth
        predicted_caption (tensor): the caption predicted by the model

    Returns:
        bleu1, bleu2, bleu3, bleu4 (float): bleu score(0-1)
    """

    reference_caption = reference_caption.squeeze(0)
    reference_caption = [caption.item() for caption in reference_caption]
    sf = SmoothingFunction()
    bleu1 = sentence_bleu([reference_caption], predicted_caption, weights = [1],
                            smoothing_function = sf.method2)
    bleu2 = sentence_bleu([reference_caption], predicted_caption, weights = [0.5, 0.5],
                            smoothing_function = sf.method2)
    bleu3 = sentence_bleu([reference_caption], predicted_caption, weights = [0.33, 0.33, 0.33], 
                            smoothing_function = sf.method2)
    bleu4 = sentence_bleu([reference_caption], predicted_caption, weights = [0.25, 0.25, 0.25, 0.25], 
                            smoothing_function = sf.method2 )
    #bleu = sentence_bleu([reference_caption], predicted_caption, smoothing_function=sf.method2)

    return bleu1, bleu2, bleu3, bleu4