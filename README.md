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
## Usage
For training:
```
python3 captioning/run_image_captioning.py -t -d <PATH_TO_DATASET> -o <OUTPUT_DIRECTORY> -p <PATH_TO_Wikipedia2Vec_EMBEDDINGS>
```
For testing:
```
python3 captioning/run_image_captioning.py -e -d <PATH_TO_DATASET> -o <OUTPUT_DIRECTORY> -p <PATH_TO_Wikipedia2Vec_EMBEDDINGS>
```
## Description
### Encoder
The encoder is ResNet-18 pretrianed on ImageNet classification dataset. The final classifcation layer has been replaced with a fully connected layer. Another fully connected layer has been added which maps fully connected layer to the required embedding size. The output of the final embedding layer is used to initialize the LSTM decoder.

During training I performed both fine-tuning and feature extraction. Fine-tuning is bit slower then feature extraction. As the Flickr-8k dataset is much smaller than ImageNet dataset and images in both datasets are similar so feature extraction is better than fine-tuning. So during feature extraction, the weights of only the newly added fully connected layers are updated. 

### Decoder
The decoder (LSTM) takes the output of final embedding layer of encoder and passes it as input to the first LSTM cell.
During training teacher forcing is used for fast convergence. In teacher forcing, instead of passing the output of previous LSTM cell to the next LSTM cell, the target word is passed as input.
