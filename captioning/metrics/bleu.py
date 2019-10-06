from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def bleu_score(reference_caption, predicted_caption):
    """
    Calculates the bleu score using nltk (one caption at a time)

    Args:
        reference_caption (tensor): ground truth
        predicted_caption (tensor): the caption predicted by the model

    Returns:
        bleu (float): bleu score(0-1)
    """

    reference_caption = reference_caption.squeeze(0)
    reference_caption = [caption.item() for caption in reference_caption]
    sf = SmoothingFunction()
    bleu = sentence_bleu([reference_caption], predicted_caption, smoothing_function=sf.method2)

    return bleu