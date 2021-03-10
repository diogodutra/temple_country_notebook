import os
import copy
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from math import ceil
import random


class ChurchesDataset(Dataset):
    """Churches dataset."""


    def __init__(self, root_dir, transform=None, *,
                  include=None, exclude=None,
                 ):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            include (list of str, optional): only include filepaths that contain
                any of these substrings
            exclude (list of str, optional): always exclude filepaths that contain
                any of these substrings
        """
        self._init_instance_variables()
        self.root_dir = root_dir
        self.transform = transform
        self.include = include
        self.exclude = exclude
        self._list_filepaths()


    def _init_instance_variables(self):
      self.filepaths = []
      self.filepaths_organized = defaultdict(lambda: [])
      self.targets = {}
      self.countries = {}
      self.counts = defaultdict(lambda: 0)


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        filepath = self.filepaths[idx]
        image = read_image(filepath)

        if self.transform:
            image = self.transform(image)
    
        country = self._get_country(filepath)

        return image, self.targets[country]


    def _list_filepaths(self):
      
      countries = set()

      # crawl files in subfolders      
      for root, dirs, files in os.walk(self.root_dir):
        for file in files:
          filepath = os.path.join(root, file)
          
          add_filepath = True
          if self.include is not None:
            add_filepath = any((_inc in filepath for _inc in self.include))
          if add_filepath:
            if self.exclude is not None:
              add_filepath = all((_exc not in filepath for _exc in self.exclude))

          if add_filepath:
            #append the file name to the list of paths
            self.filepaths.append(filepath)
            
            # add country to the set of know countries
            country = self._get_country(filepath)
            countries.add(country)
            self.counts[country] += 1
            self.filepaths_organized[country].append(filepath)

          
      # append the country to the targets dictionary
      for i, country in enumerate(sorted(countries)):
          self.countries[i] = country
          self.targets[country] = i

      self.counts = dict(self.counts) # remove defauldict
      self.filepaths_organized = dict(self.filepaths_organized) # remove defauldict


    def _get_country(self, filepath):
      return filepath.split('/')[-2]


    def plot_occurrencies(self):
        """
        Plots the occurrencies of each label in dataset.
        """
        n_countries = len(self.countries)
        countries = [self.countries[i] for i in range(n_countries)]
        occurrencies = [self.counts[c] for c in countries]
        plt.bar(countries, occurrencies)
        plt.gcf().patch.set_facecolor('white')
        plt.ylabel('occurrencies')
        plt.title('Dataset')
        plt.xticks(rotation=90)
        plt.gca().set_axisbelow(True)
        plt.gca().yaxis.grid(color='gray', linestyle='dashed')


    def plot_church(self, idx=None):
        """
        Plots the image of a church.

        Args:
            idx (int, default random): index of the church from the dataset.filepaths list
        """
        if idx is None: idx = np.random.choice(range(len(self)))

        img, tgt = self[idx]
        
        img = to_numpy(img)

        _ = plt.imshow(img)
        filepath = '/'.join(self.filepaths[idx].split('/')[-2:])
        height, width, channels = img.shape
        title = filepath + f' ({width} x {height})'
        plt.gcf().patch.set_facecolor('white')
        plt.title(title)


    def plot_booth(self, indices=None, predictions=None, *,
                   cols=4, rows=2):
        """
        Plots many small images of the same church.

        Args:
            indices (list of int, default random repeated): indices for
                dataset.filepaths for every church to be plotted
            predictions (list of int, optional): classifier predictions for
                every index in indices, to remark hits and misses
            cols (int, default 4): columns of images
            rows (int, default 2): rows of images if indices is omitted
        """
        prediction_colors = {True: 'g', False: 'r'}
        
        if indices is None:
            idx = np.random.choice(range(len(self)))
            indices = [idx] * cols * rows # repeat the same index
        else:
            rows = ceil(len(indices) / cols)

        for i, index in enumerate(indices):
            plt.subplot(rows, cols, i+1)
            img, tgt = self[index]
            self.plot_church(index)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)

            if predictions is None:
                plt.gca().set_frame_on(False)
                plt.title(f'true:{tgt}')
            else:
                prediction = int(predictions[index])
                plt.title(f'true:{tgt} pred:{prediction}')
                correct = (prediction == tgt)
                plt.setp(plt.gca().spines.values(), linewidth=2,
                         color=prediction_colors[correct])
                

def read_image(image_path):
    """Returns NumPy array from image filename, ignoring EXIF orientation."""
    r = open(image_path,'rb').read()
    img_array = np.asarray(bytearray(r), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)[...,::-1]
    return img.copy()


def to_numpy(img):
    """
    Converts image from torch.Tensor type to NumPy array object for image plot.
    """
    if type(img) == torch.Tensor:
      img = (img * 255).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    
    return img


def stratified_split(dataset, k=1, *, shuffle=False):
    """
    Splits the dataset ensuring k samples of each country.
    
    Args:
        dataset (ChurchesDataset): dataset with all the churches images.
        k (int, default 1): occurrencies of each country

    Returns:
        dataset_split (ChurchesDataset): splitted dataset with k samples for each country
        dataset_left (ChurchesDataset): left-over from the input dataset after split
        shuffle (bool, default False): randomly selects k images from every country
    """
    dataset_split = copy.deepcopy(dataset)
    dataset_split.filepaths = []
    dataset_left = copy.deepcopy(dataset)
    dataset_left.filepaths = []

    for target, target_filepaths in dataset.filepaths_organized.items():

        indices_all = range(len(target_filepaths))

        if shuffle:
            indices_split = random.sample(indices_all, k)
        else:
            indices_split = range(k)

        indices_left = [i for i in indices_all if i not in indices_split]

        target_filepaths_split = [target_filepaths[i] for i in indices_split]
        target_filepaths_left  = [target_filepaths[i] for i in indices_left]

        # dataset_split
        dataset_split.filepaths_organized[target] = target_filepaths_split
        dataset_split.filepaths.extend(target_filepaths_split)
        dataset_split.counts[target] = len(target_filepaths_split)
        
        # dataset_left
        dataset_left.filepaths_organized[target] = target_filepaths_left
        dataset_left.filepaths.extend(target_filepaths_left)
        dataset_left.counts[target] = len(target_filepaths_left)

    return dataset_split, dataset_left