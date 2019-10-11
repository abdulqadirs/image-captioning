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
import logging

from config import Config

logger = logging.getLogger('captioning')
                                                                
def read_captions(path):
    """
    Reads the captions of images, strips and appends <start> & <end> tokens.

    Args:
        path (Path): Path of the file containing captions of images.

    Returns:
        A dictionary of {image_ids: list of captions}
    
    Raises:
        FileNotFoundError: If the file containing captions doesn't exist.
    """
    try:
        raw_captions_file = open(path, 'r').read().strip().split('\n')
    except FileNotFoundError:
        logger.exception("Traceback of captions file '{}' not found.".format(path))
    else:
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
    Constructs the dictionary of words in captions based on frequency of each word.
    
    Args:
        raw_captions (dict): Dictionary of {image_ids: [image captions]}.
        threshold (int): Words from image captions are being included in dictionary if frequency of words >= threshold.

    Returns:
            id_to_word (list): list of words in dictionary indexed by id.
            word_to_id (dict): dictionary of {word: id}.
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

    Args:
        raw_captions (dict): A dictionary of {image_ids: list of captions}.
        word_to_id (dict): A dictionary of {word: id}.

    Returns:
        captions(list): List of ids of word from dictionary.
        image ids(str): Name of image files without extension.
        lengths(list): Actual length of each caption.
    """
    tokens, image_ids = [], []
    #maxlen = 20
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
            #if len(token) > maxlen:
            #    token = token[:maxlen]
            #else:
            #    token += [0] * (maxlen-len(token))
            tokens.append(token)
            image_ids.append(img_id)
    #tokens = np.array(tokens).astype('int32')
    image_ids = np.array(image_ids)
    
    return tokens, image_ids, lengths


class PadSequence(object):
    """
    Pads the unequal sequences with zeros in a batch to make them equal to the largest sequence in the batch.
    Also sorts them in descending order.
    """
    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        lengths = [item[2] for item in batch]
        image_ids = [item[3] for item in batch]
        max_length = max(lengths)
        if len(lengths) > 1:
            for caption in captions:
                if len(caption) < max_length:
                    pad = [0] * (max_length - len(caption))
                    caption += pad
        
            images, captions, lengths = self.sort(images, captions, lengths)
        return torch.stack(images), torch.LongTensor(captions), lengths, image_ids
    
    def sort(self, images, padded_captions, lengths):
        """
        Sorts in 'lengths' of captions in descending order.
        Then sorts the images and captions according to the sorted 'lengths'.

        Args:
            images (list of tensors):
            padded_captions (list):
            lengths (list):
        
        Returns:
            Sorted lengths, images and captions in descending order according to the lengths of captions.
        """
    
        lengths = torch.tensor(lengths)
        lengths, indices = torch.sort(lengths, descending=True)
        sorted_captions = []
        sorted_images = []
        for index in indices:
            sorted_captions.append(padded_captions[index])
            sorted_images.append(images[index])
        return sorted_images, sorted_captions, lengths
  

class Flickr8k(Dataset):
    """
    Loads the dataset.
    """
    def __init__(self,captions_path, dataset_directory, transform):
        self.dataset_directory = dataset_directory
        self.captions_path = captions_path
        self.raw_captions = read_captions(self.captions_path)
        self.id_to_word, self.word_to_id = dictionary(self.raw_captions, threshold = 5)
        self.captions, self.image_ids, self.lengths = tokenization(self.raw_captions, self.word_to_id)
        self.transform = transform

    def __getitem__(self, index):
        """
        Reads images and returns data.
        Applies the transformations.
        
        Returns:
            im (tensor): Images
            caption (tensor): Captions
            lengths (list): Actual lengths of captions.
            im_id (list): Name of image files without extension.
        
        Raises:
            FileNotFoundError: If the images directory path is invalid.
        """
        # TODO (aq): Raise error if the files doesn't exist.
        caption = self.captions[index]
        im_id = self.image_ids[index]
        lengths = self.lengths[index]
        im = Image.open(self.dataset_directory / 'Flickr8k_images' /   im_id)
        if self.transform is not None:
            im = self.transform(im)
            
        return im, caption, lengths, im_id

    def __len__(self):
        return len(self.image_ids)


def data_loaders(images_path, captions_path):
    """
    Loads the data and divides it into training, validation and test sets 
    using samples and Flickr8k Class.

    Args:
        images_path (Path): Path of images.
        captions_path (Path): Path of captions file.

    Returns:
        training_data_loader
        validation_data_loader
        testing_data_loader
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
    
    collate = PadSequence()

    training_loader = DataLoader(flickr8k_dataset,
                        num_workers = 1,
                        batch_size = Config.get("training_batch_size"),
                        sampler = train_sampler,
                        collate_fn = collate)

    validation_loader = DataLoader(flickr8k_dataset,
                        num_workers = 1,
                        batch_size = Config.get("validation_batch_size"),
                        sampler = valid_sampler,
                        collate_fn = collate)

    testing_loader = DataLoader(flickr8k_dataset,
                        num_workers = 1,
                        batch_size = Config.get("testing_batch_size"),
                        sampler= test_sampler,
                        collate_fn = collate)

    return training_loader, validation_loader, testing_loader