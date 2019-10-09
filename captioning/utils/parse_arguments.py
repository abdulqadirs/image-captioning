from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

def parse_arguments():
    """
    Parses the arguments passed through the terminal.

    Returns:
        The arguments passed through the terminal.
    """
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', '--dataset', help='Dataset directory', required=True)
    parser.add_argument('-o', '--output_directory', help='Directory where to output artifacts', required=True)
    parser.add_argument('-p', '--pretrained_embeddings', 
                        help='Directory where the pretrained embeddings are stored', required=True)
    #train or test the model
    mode_parser = parser.add_mutually_exclusive_group(required=True)
    mode_parser.add_argument('-t', '--training', help='Train the model.', action='store_true')
    mode_parser.add_argument('-e', '--testing', help='Test the model using stored checkpoint if exists.',
                            action='store_true')

    return parser.parse_known_args()