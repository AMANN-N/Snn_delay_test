#from utils import set_seed

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Callable, Optional
#from tensorflow import set_random_seed
import tensorflow as tf

import torchvision.transforms as transforms

#from spikingjelly.datasets.shd import SpikingHeidelbergDigits
#from spikingjelly.datasets.shd import SpikingSpeechCommands
#from spikingjelly.datasets import pad_sequence_collate
#from utils import set_seed

import numpy as np
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torch
import torch.nn as nn
import requests
import pandas as pd
import zipfile
import os
import urllib.request
import tarfile
import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Callable, Optional

import torchvision.transforms as transforms

#from spikingjelly.datasets.shd import SpikingHeidelbergDigits
#from spikingjelly.datasets.shd import SpikingSpeechCommands
#from spikingjelly.datasets import pad_sequence_collate



import torch
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, Resample
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchvision import transforms
from torch.utils.data import Dataset


class RNoise(object):

  def __init__(self, sig):
    self.sig = sig

  def __call__(self, sample):
    noise = np.abs(np.random.normal(0, self.sig, size=sample.shape).round())
    return sample + noise


class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config


  def __call__(self, x, y):
    # Sample shape: (time, neurons)
    for sample in x:
      # Time mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.time_mask_size)
        ind = np.random.randint(0, sample.shape[0] - self.config.time_mask_size)
        sample[ind:ind+mask_size, :] = 0

      # Neuron mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.neuron_mask_size)
        ind = np.random.randint(0, sample.shape[1] - self.config.neuron_mask_size)
        sample[:, ind:ind+mask_size] = 0

    return x, y


class CutMix(object):
  """
  Apply Spectrogram-CutMix augmentaiton which only cuts patch across time axis unlike
  typical Computer-Vision CutMix. Applies CutMix to one batch and its shifted version.

  """

  def __init__(self, config):
    self.config = config


  def __call__(self, x, y):

    # x shape: (batch, time, neurons)
    # Go to L-1, no need to augment last sample in batch (for ease of coding)

    for i in range(x.shape[0]-1):
      # other sample to cut from
      j = i+1

      if np.random.uniform() < self.config.cutmix_aug_proba:
        lam = np.random.uniform()
        cut_size = int(lam * x[j].shape[0])

        ind = np.random.randint(0, x[i].shape[0] - cut_size)

        x[i][ind:ind+cut_size, :] = x[j][ind:ind+cut_size, :]

        y[i] = (1-lam) * y[i] + lam * y[j]

    return x, y



class Augs(object):

  def __init__(self, config):
    self.config = config
    self.augs = [TimeNeurons_mask_aug(config), CutMix(config)]

  def __call__(self, x, y):
    for aug in self.augs:
      x, y = aug(x, y)

    return x, y

"""

def SHD_dataloaders(config):
  set_seed(config.seed)

  train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  #train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  #valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, test_loader




def SSC_dataloaders(config):
  set_seed(config.seed)

  train_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='train', data_type='frame', duration=config.time_step)
  valid_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='valid', data_type='frame', duration=config.time_step)
  test_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='test', data_type='frame', duration=config.time_step)


  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader



class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label

class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
    def __init__(
            self,
            root: str,
            n_bins: int,
            split: str = 'train',
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        
        The Spiking Speech Commands (SSC) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        
        super().__init__(root, split, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label



"""





































#import sys
#sys.path.append('/home/amanrs/PaSST/')
#from models.preprocess import AugmentMelSTFT
# Now you can import functions from files in that directory
#from external_file import function_name













#labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


url = 'https://github.com/karolpiczak/ESC-50/archive/master.zip'
filename = 'ESC-50-master.zip'
target_dir = './'


#labels = ['Dog' ,'Rain','Crying baby','Door knock',	'Helicopter','Rooster',	'Sea waves'	,'Sneezing','Mouse click','Chainsaw','Pig',	'Crackling fire','Clapping'	,'Keyboard typing'	,'Siren','Cow',	'Crickets',	'Breathing'	,'Door wood creaks','Car horn','Frog','Chirping birds','Coughing',	'Can opening'	,'Engine','Cat'	,'Water drops'	,'Footsteps',	'Washing machine'	,'Train','Hen'	,'Wind'	,'Laughing'	,'Vacuum cleaner',	'Church bells','Insects (flying)',	'Pouring water'	,'Brushing teeth'	,'Clock alarm',	'Airplane''Sheep',	'Toilet flush'	,'Snoring'	,'Clock tick'	,'Fireworks','Crow',	'Thunderstorm'	,'Drinking sipping'	,'Glass breaking'	,'Hand saw']
#labels = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow","rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush","thunderstorm", "baby_crying", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring","drinking_sip", "door_knock", "mouse_click", "keyboard_typing", "door_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_tick","glass_breaking", "helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks"]
#labels = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow','rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush','thunderstorm', 'baby_crying', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth', 'snoring','drinking_sip', 'door_knock', 'mouse_click', 'keyboard_typing', 'door_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_tick','glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks']
labels = list(range(50))


def target_transform_fn(word):
    try:
        index = labels.index(word)
        return torch.tensor(index)
    except ValueError:
        print(f"Label '{word}' not found in labels list")
        return None



def download_and_extract_dataset(url, filename, target_dir):
    urllib.request.urlretrieve(url, filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    os.rename(os.path.join(target_dir, 'ESC-50-master'), os.path.join(target_dir, 'esc50_dataset'))

  

def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.2, random_seed=42):
    meta_data = pd.read_csv(os.path.join(data_path, 'meta/esc50.csv'))
    
    all_filenames = [os.path.join(data_path, 'audio_32k', filename) for filename in meta_data['filename']]
    all_labels = meta_data['target'].values
    
    train_filenames, test_filenames, train_labels, test_labels = train_test_split(all_filenames, all_labels, test_size=test_size, random_state=random_seed)
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(train_filenames, train_labels, test_size=val_size, random_state=random_seed)
    
    return train_filenames, val_filenames, test_filenames, train_labels, val_labels, test_labels



class PadOrTruncate(object):
    """Pad all audio to specific length."""
    def __init__(self, audio_length):
        self.audio_length = audio_length

    def __call__(self, sample):
        #print(self.audio_length) = 220500
        #print(sample.size(-1)) = 220500
        if len(sample) <= self.audio_length:
            return F.pad(sample, (0, self.audio_length - sample.size(-1)))
        else:
            return sample[0: self.audio_length]
        
        
        
    def __repr__(self):
        return f"PadOrTruncate(audio_length={self.audio_length})"
        


#this function is being passed a tensor which represents a waveform, sample rate of the audio is 44100 and the audio is 5 seconds long
def build_transform(is_train):
    sample_rate= 32000       #figure out if this should be 44100 or 220500 and below *5
    window_size= 320
    hop_length= 220
    n_mels= 160
    f_min= 50
    f_max= 14000

    t = [PadOrTruncate(sample_rate*5),
         Resample(sample_rate, sample_rate // 2)]
    
    #n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
    #                            timem=80,
    #                             htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
    #                             fmax_aug_range=2000

    # now the waveform is at sample_rate = 32k and length is 5
    t.append(Spectrogram(n_fft=window_size, hop_length=hop_length, power=2))
    #t.append(AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
    #                             timem=80,
    #                             htk=False, fmin=50, fmax=14000, norm=1, fmin_aug_range=10,
    #                             fmax_aug_range=2000))


    if is_train:
        pass

    t.extend([MelScale(n_mels=n_mels,
                       sample_rate=sample_rate // 2,
                       f_min=f_min,
                       f_max=f_max,
                       n_stft=window_size // 2 + 1),
              AmplitudeToDB()
             ])

    return transforms.Compose(t)








class ESC50Dataset(Dataset):
    def __init__(self ,  dataset , targets ,  split_name, transform, target_transform ):
        
        self.dataset = dataset
        self.targets = targets
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform



    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):

        audio_path = self.dataset[index]
        #print(audio_path)
        
        label = self.targets[index]
        #print(label)
        try:
            #print(f"Loading audio file: {audio_path}")
            waveform, _ = torchaudio.load(audio_path)
            #print(waveform.shape)       #= 1 x 16k
        except Exception as e:
            #print(f"Error loading audio file '{audio_path}': {e}")
            return None, None

        if self.transform is not None:
            waveform = self.transform(waveform).squeeze().t()
            #print(waveform.shape)
        #print(waveform.shape) = 364 x 160

        target = label

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        #print(waveform.size(-1)) = 160
        #print(waveform.shape)
        #print(target.shape)
        #print(type(target))

        
        return waveform, target, torch.zeros(1)



def ESC_50(config):
  #set_seed(config.seed)
  #set_random_seed()
  tf.random.set_seed(config.seed)
  
 


  #download_and_extract_dataset(url, filename, target_dir)
  #data_path = 'esc50_dataset/'
  data_path = '/home/amanrs/PaSST/audioset_hdf5s/esc50'
  # Load and preprocess data
  train_filenames, val_filenames, test_filenames, train_labels, val_labels, test_labels = load_and_preprocess_data(data_path)
  # Define transforms for data augmentation


  train_dataset = ESC50Dataset(train_filenames, train_labels,'training', transform=build_transform(True), target_transform=target_transform_fn)
  valid_dataset = ESC50Dataset(val_filenames,val_labels ,'validation', transform=build_transform(True), target_transform=target_transform_fn)
  test_dataset = ESC50Dataset(test_filenames,test_labels ,'testing', transform=build_transform(True), target_transform=target_transform_fn)


  #print(test_dataset.size())
  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader




























class ESC50Dataset2(Dataset):
    def __init__(self ,  dataset , targets ,  split_name, transform, target_transform ):
        
        self.dataset = dataset
        self.targets = targets
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform



    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):

        audio_path = self.dataset[index]
        #print(audio_path)
        
        label = self.targets[index]
        #print(label)
        try:
            #print(f"Loading audio file: {audio_path}")
            waveform, _ = torchaudio.load(audio_path)
            #print(waveform.shape) = 1 x 16k
        except Exception as e:
            #print(f"Error loading audio file '{audio_path}': {e}")
            return None, None

        #if self.transform is not None:
        #    waveform = self.transform(waveform).squeeze().t()
            #print(waveform.shape)
        #print(waveform.shape) = 364 x 160
        #print(waveform.shape)

        target = label

        #print(target.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        #print(waveform.size(-1)) = 160
        #print(waveform.shape)
        #print(target.shape)
        #print(type(target))

        
        return waveform, target, torch.zeros(1)



def ESC_502(config):
  #set_seed(config.seed)
  #set_random_seed()
  tf.random.set_seed(config.seed)
  

  #download_and_extract_dataset(url, filename, target_dir)
  #data_path = 'esc50_dataset/'
  data_path = '/home/amanrs/PaSST/audioset_hdf5s/esc50'
  # Load and preprocess data
  train_filenames, val_filenames, test_filenames, train_labels, val_labels, test_labels = load_and_preprocess_data(data_path)
  # Define transforms for data augmentation


  train_dataset = ESC50Dataset2(train_filenames, train_labels,'training', transform=build_transform(False), target_transform=target_transform_fn)
  valid_dataset = ESC50Dataset2(val_filenames,val_labels ,'validation', transform=build_transform(False), target_transform=target_transform_fn)
  test_dataset = ESC50Dataset2(test_filenames,test_labels ,'testing', transform=build_transform(False), target_transform=target_transform_fn)


  #print(test_dataset.size())
  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)
  #print(test_loader.size())

  #for i, (x, y, _) in enumerate(valid_loader):
  #  print(x.shape)
  #  print(y.shape)

  return train_loader, valid_loader, test_loader
