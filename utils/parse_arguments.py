from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

def parse_arguments():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', '--dataset', help='Dataset directory', required=True)
    parser.add_argument('-o', '--output_directory', help='Directory where to output artifacts', required=True)
    parser.add_argument('-e', '--pretrained_embeddings', 
                        help='Directory where the pretrained embeddings are stored', required=True)

    return parser.parse_args()