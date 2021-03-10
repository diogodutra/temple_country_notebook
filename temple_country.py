
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_boilerplate.transforms as my_transforms
from pytorch_boilerplate.trainer import (
  TrainerClassifier, TransferLearning,
)
from dataset.churches_dataset import ChurchesDataset


def load_model(model_filename, num_countries=11):
  model = TransferLearning(num_target_classes = num_countries)

  # load model parameters
  model_state_dict = torch.load(model_filename)
  model.load_state_dict(model_state_dict)
  _ = model.eval()

  return model


def load_dataset(folder):
  sacred_class_weight = {
  # class_id: class_weight
      497: 1, # church
      668: 1, # mosque
      663: 1, # monastery
      832: 1, # stupa
      }

  transform = transforms.Compose([
    transforms.ToTensor(),
    my_transforms.BestSquareCrop(sacred_class_weight),
    transforms.Resize((224, 224)),
    ])

  return ChurchesDataset(folder, transform)


def index_to_country(list_indices):
  country_dict = {
    0: 'Armenia',
    1: 'Australia',
    2: 'Germany',
    3: 'Hungary+Slovakia+Croatia',
    4: 'Indonesia-Bali',
    5: 'Japan',
    6: 'Malaysia+Indonesia',
    7: 'Portugal+Brazil',
    8: 'Russia',
    9: 'Spain',
    10: 'Thailand',
    }
  return [country_dict[i] for i in list_indices]


def main(
  folder = '.',
  model = '/model/model.pth',
  output = 'results.csv',
  batch = 4,
  ):

  model = load_model(model)

  dataset = load_dataset(folder)
  print(len(dataset), 'images found.')
  
  loader = DataLoader(dataset, batch_size=min(batch, len(dataset)))
  
  trainer = TrainerClassifier(model=model)
  y_pred = trainer.predict(loader)
  countries = index_to_country(y_pred)

  # extract features
  df = pd.DataFrame(data=np.c_[dataset.filepaths, countries],
   columns=['filename', 'country'])

  df.to_csv(output)

  print('Countries saved on file', output)

  return df


if __name__ == '__main__':

  # parse script parameters
  parser = argparse.ArgumentParser(
    description="Guess the country of the temples in all pictures.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-f", "--folder", default=".", help="path containing pictures of temples")
  parser.add_argument("-o", "--output", default="results.csv", help="filename that will contain the predictions")
  parser.add_argument("-m", "--model", default="/model/model.pth", help="filename with the model state parameters")
  parser.add_argument("-b", "--batch", default=4, help="batch size for torch.DataLoader")
  args = parser.parse_args()

  # Create a dictionary of the shell arguments
  kwargs = vars(args)
  
  # run prediction pipeline as main script
  main(**kwargs)
