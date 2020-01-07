# Image Captioning
Implementation of image captioning using CNN encoder and LSTM decoder on Flickr8k dataset.

## Setup
Install the requirements:
```
pip install -r requirements.txt
```
Then download the pretrained
[Wikipedia2Vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) 
emebeddings.
### Usage
For training and validation:
```
python3 captioning/run_image_captioning.py -t -d <PATH_TO_DATASET> -o <OUTPUT_DIRECTORY> -p <PATH_TO_Wikipedia2Vec_EMBEDDINGS>
```
For testing:
```
python3 captioning/run_image_captioning.py -e -d <PATH_TO_DATASET> -o <OUTPUT_DIRECTORY> -p <PATH_TO_Wikipedia2Vec_EMBEDDINGS>
```