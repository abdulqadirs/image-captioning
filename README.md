# Image Captioning
Image Captioning

Implementation of image captioning using CNN encoder and LSTM decoder.

## Prerequisites
* Python 3.6
* PyTorch 1.2
* NumPy
* tqdm
* TensorBoard
* gensim
* NLTK
* Word2vec embeddings

### Usage
For training and validation:
```
python3 captioning/run_image_captioning.py -t -d <PATH_TO_DATASET> -o <OUTPUT_DIRECTORY> -p <PATH_TO_WORD2VEC_EMBEDDINGS>
```
For testing:
```
python3 captioning/run_image_captioning.py -e -d <PATH_TO_DATASET> -o <OUTPUT_DIRECTORY> -p <PATH_TO_WORD2VEC_EMBEDDINGS>
```