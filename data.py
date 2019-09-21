import nltk
import itertools
import numpy as np
import torch.utils.data as utils
from PIL import Image
import torch
import PIL.ImageOps
from torch.utils.data import Dataset, DataLoader  
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


                                                                
def read_captions(path):
    """
    Reads the captions of images, strips and appends <start> & <end> tokens

    Params
    -----
    - path: path of the file containing captions of images

    Returns
    -------
    - a dictionary of {image_ids: list of captions}
    """
    raw_captions_file = open(path, 'r').read().strip().split('\n')
    raw_captions = {}
    for line in raw_captions_file:
        line = line.strip(' .')
        line = line.split('\t')
        img_id, caption = line[0][:len(line[0])-2], line[1]
        if img_id not in raw_captions:
            raw_captions[img_id] = ['<start> ' + caption + ' <end>']
        else:
            raw_captions[img_id].append('<start> ' + caption + ' <end>')
    return raw_captions


def dictionary(raw_captions,threshold):
    """
    Constructs the dictionary of words in captions based on frequency of each word
    
    Params
    ------
    - raw_captions: dictionary of {image_ids: [image captions]}
    - threshold: words from image captions are included in dictionary if frequency of words >= threshold

    Returns
    -------
    - list of words in dictionary indexed by id
    - dictionary of {word: id}
    """
    caps = []
    for im in raw_captions:
        for s in raw_captions[im]:
            caps.append(s.split())

    word_freq = nltk.FreqDist(itertools.chain(*caps))
    id_to_word = ['<pad>'] + [word for word, cnt in word_freq.items() if cnt >= threshold] + ['<unk>']
    word_to_id = {word:idx for idx, word in enumerate(id_to_word)}
    
    return id_to_word, word_to_id


def tokenization(raw_captions, word_to_id):
    """
    Represents the raw captions by list of ids in the dictionary

    Params
    ------
    - raw_captions: a dictionary of {image_ids: list of captions}
    - word_to_id: dictionary of {word: id}

    Returns
    -------
    - captions (list of ids of word from dictionary)
    - image ids
    - actual length of each caption
    """
    tokens, image_ids = [], []
    maxlen = 20
    lengths = []
    for img_id in raw_captions:
        for cap in raw_captions[img_id]:
            token = []
            for word in cap.split():
                if word in word_to_id:
                    token.append(word_to_id[word])
                else:
                    token.append(word_to_id['<unk>'])
            lengths.append(len(token))
            if len(token) > maxlen:
                token = token[:maxlen]
            else:
                token += [0] * (maxlen-len(token))
            tokens.append(token)
            image_ids.append(img_id)
    tokens = np.array(tokens).astype('int32')
    image_ids = np.array(image_ids)
    
    return tokens, image_ids, lengths


class Flickr8k(Dataset):
    def __init__(self,captions_path, dataset_directory, transform):
        self.dataset_directory = dataset_directory
        self.captions_path = captions_path
        self.raw_captions = read_captions(self.captions_path)
        self.id_to_word, self.word_to_id = dictionary(self.raw_captions, threshold = 5)
        self.captions, self.image_ids, self.lengths = tokenization(self.raw_captions, self.word_to_id)
        self.transform = transform

    def __getitem__(self, index):
        caption = self.captions[index]
        im_id = self.image_ids[index]
        lengths = self.lengths[index]
        im = Image.open('../Flickr8k_Dataset/Flickr8k_images/Flickr8k_images/' + im_id)
        if self.transform is not None:
            im = self.transform(im)
            
        return im, torch.LongTensor(caption), lengths

    def __len__(self):
        return len(self.image_ids)


def data_loaders(images_path, captions_path):
    """
    Loads the data and divides it into training, validation and test sets using samples

    Params
    ------
    - images_path:
    - captions_path:

    Returns
    -------
    - training data loader
    - validation data loader
    - testing data loader


    """
    images_dir = images_path
    dataset_folder = datasets.ImageFolder(root = images_dir)
    captions_path = captions_path
    flickr8k_dataset = Flickr8k(captions_path, images_dir, transform = transforms.Compose([transforms.Resize((224,224)),
                                                                        transforms.ToTensor()]))

    dataset_size = len(dataset_folder)
    indices = list(range(dataset_size))
    training_split = int(0.8 * dataset_size)
    validation_split = int(0.9 * dataset_size)

    np.random.seed(96)
    np.random.shuffle(indices)

    train_indices = indices[:training_split]
    valid_indices = indices[training_split:validation_split]
    test_indices = indices[validation_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    training_loader = DataLoader(flickr8k_dataset,
                        num_workers = 1,
                        batch_size = 2,
                        sampler= train_sampler)

    validation_loader = DataLoader(flickr8k_dataset,
                        num_workers = 1,
                        batch_size = 2,
                        sampler= valid_sampler)

    testing_loader = DataLoader(flickr8k_dataset,
                        num_workers = 1,
                        batch_size = 2,
                        sampler= test_sampler)

    return training_loader, validation_loader, testing_loader