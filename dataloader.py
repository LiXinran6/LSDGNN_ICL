from dataset import *
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from  transformers import BertTokenizer

def get_train_valid_sampler(trainset):
    size = len(trainset)
    idx = list(range(size))
    return SubsetRandomSampler(idx)


def load_vocab(dataset_name):
    speaker_vocab = pickle.load(open('data/%s/speaker_vocab.pkl' % (dataset_name), 'rb'))
    label_vocab = pickle.load(open('data/%s/label_vocab.pkl' % (dataset_name), 'rb'))
    person_vec_dir = 'data/%s/person_vect.pkl' % (dataset_name)
    # if os.path.exists(person_vec_dir):
    #     print('Load person vec from ' + person_vec_dir)
    #     person_vec = pickle.load(open(person_vec_dir, 'rb'))
    # else:
    #     print('Creating personality vectors')
    #     person_vec = np.random.randn(len(speaker_vocab['itos']), 100)a
    #     print('Saving personality vectors to' + person_vec_dir)
    #     with open(person_vec_dir,'wb') as f:
    #         pickle.dump(person_vec, f, -1)
    person_vec = None

    return speaker_vocab, label_vocab, person_vec


def get_IEMOCAP_loaders(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    devset = IEMOCAPDataset(dataset_name, 'dev', speaker_vocab, label_vocab, args)
    valid_sampler = get_train_valid_sampler(devset)
    testset = IEMOCAPDataset(dataset_name, 'test',  speaker_vocab, label_vocab, args)
    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return valid_loader, test_loader, speaker_vocab, label_vocab, person_vec
#adding babystep_index argument if using curriculum learning
def get_train_loader(dataset_name = 'IEMOCAP', batch_size=32, num_workers=0, pin_memory=False, args = None, babystep_index = None):
    print('building vocab.. ')
    speaker_vocab, label_vocab, person_vec = load_vocab(dataset_name)
    print('building datasets..')
    if (args.curriculum):
            trainset = IEMOCAPDataset(dataset_name, 'train', speaker_vocab, label_vocab, args, None, babystep_index)
            train_sampler = get_train_valid_sampler(trainset)
            train_loader = DataLoader(trainset,
                                      batch_size=batch_size,
                                      sampler=train_sampler,
                                      collate_fn=trainset.collate_fn,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
            # train_loaders.append(train_loader)
    else:
        trainset = IEMOCAPDataset(dataset_name, 'train', speaker_vocab, label_vocab, args)
        train_sampler = get_train_valid_sampler(trainset)
        train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
        # train_loaders.append(train_loader)

    return train_loader