# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk

import itertools
import numpy

from six.moves import range
from itertools import chain
from scipy import fft
from scipy.signal import stft


FS = 44100            # samples/second
DEFAULT_WINDOW_SIZE = 2048    # fourier window size
OUTPUT_SIZE = 128               # number of distinct notes
STRIDE = 512          # samples between windows
WPS = FS / float(512)   # windows/second


class MusicNet(object):
    def __init__(self, filename, in_memory=True, window_size=4096,
                 output_size=84, feature_size=1024, sample_freq=11000,
                 complex_=False, fourier=False, stft=False, fast_load=False,
                 rng=None, seed=123):
        if not in_memory:
            raise NotImplementedError
        self.filename = filename

        self.window_size = window_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.sample_freq = sample_freq
        self.complex_ = complex_
        self.fourier = fourier
        self.stft = stft
        self.fast_load = fast_load

        if rng is not None:
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(seed)

        self._train_data = {}
        self._valid_data = {}
        self._test_data = {}
        self._loaded = False

        self._eval_sets = {}

    def splits(self):
        with open(self.filename, 'rb') as f:
            # This should be fast
            all_inds = numpy.load(f).keys()
        test_inds = ['2303', '2382', '1819']
        valid_inds = ['2131', '2384', '1792',
                      '2514', '2567', '1876']
        train_inds = [ind for ind in all_inds
                      if ind not in test_inds and ind not in test_inds]
        return train_inds, valid_inds, test_inds

    @classmethod
    def note_to_class(cls, note):
        return note - 21

    @property
    def train_data(self):
        if self._train_data == {}:
            self.load()
        return self._train_data

    @property
    def valid_data(self):
        if self._valid_data == {}:
            self.load()
        return self._valid_data

    @property
    def test_data(self):
        if self._test_data == {}:
            self.load()
        return self._test_data

    def load(self, filename=None, reload=False):
        if filename is None:
            filename = self.filename
        if self._loaded and not reload:
            return

        with open(filename, 'rb') as f:
            train_inds, valid_inds, test_inds = self.splits()
            data_file = numpy.load(f)
            if self.fast_load:
                train_inds = train_inds[:6]
                train_data = {}
                for ind in chain(train_inds, valid_inds, test_inds):
                    train_data[ind] = data_file[ind]
            else:
                train_data = dict(data_file)

            # test set
            test_data = {}
            for ind in test_inds:
                if ind in train_data:
                    test_data[ind] = train_data.pop(ind)

            # valid set
            valid_data = {}
            for ind in valid_inds:
                valid_data[ind] = train_data.pop(ind)

            self._train_data = train_data
            self._valid_data = valid_data
            self._test_data = test_data

    def construct_eval_set(self, data, step=128):
        n_files = len(data)
        pos_per_file = 7500
        features = numpy.empty([n_files * pos_per_file, self.window_size])
        outputs = numpy.zeros([n_files * pos_per_file, self.output_size])

        features_ind = 0
        labels_ind = 1

        for i, ind in enumerate(data):
            print(ind)
            audio = data[ind][features_ind]

            for j in range(pos_per_file):
                if j % 1000 == 0:
                    print(j)
                # start from one second to give us some wiggle room for larger
                # segments
                index = self.sample_freq + j * step
                features[pos_per_file * i + j] = audio[index:
                                                       index + self.window_size]

                # label stuff that's on in the center of the window
                s = int((index + self.window_size / 2))
                for label in data[ind][labels_ind][s]:
                    note = label.data[1]
                    outputs[pos_per_file * i + j, self.note_to_class(note)] = 1
        return features, outputs

    @property
    def feature_dim(self):
        dummy_features = numpy.zeros((1, self.window_size, 1))
        dummy_output = numpy.zeros((1, self.output_size))
        dummy_features, _ = self.aggregate_raw_batch(
            dummy_features, dummy_output)
        return dummy_features.shape[1:]

    def aggregate_raw_batch(self, features, output):
        """Aggregate batch.

        All post processing goes here.

        Parameters:
        -----------
        features : 3D float tensor
            Input tensor
        output : 2D integer tensor
            Output classes

        """
        channels = 2 if self.complex_ else 1
        features_out = numpy.zeros(
            [features.shape[0], self.window_size, channels])
        if self.fourier:
            if self.complex_:
                data = fft(features, axis=1)
                features_out[:, :, 0] = numpy.real(data[:, :, 0])
                features_out[:, :, 1] = numpy.imag(data[:, :, 0])
            else:
                data = numpy.abs(fft(features, axis=1))
                features_out = data
        elif self.stft:
            _, _, data = stft(features, nperseg=120, noverlap=60, axis=1)
            length = data.shape[1]
            n_feats = data.shape[3]
            if self.complex_:
                features_out = numpy.zeros(
                    [len(self.train_data), length, n_feats * 2])
                features_out[:, :, :n_feats] = numpy.real(data)
                features_out[:, :, n_feats:] = numpy.imag(data)
            else:
                features_out = numpy.abs(data[:, :, 0, :])
        else:
            features_out = features
        return features_out, output

    def train_iterator(self):
        features = numpy.zeros([len(self.train_data), self.window_size])

        while True:
            output = numpy.zeros([len(self.train_data), self.output_size])
            for j, ind in enumerate(self.train_data):
                s = self.rng.randint(
                    self.window_size / 2,
                    len(self.train_data[ind][0]) - self.window_size / 2)
                data = self.train_data[ind][0][s - self.window_size / 2:
                                               s + self.window_size / 2]
                features[j, :] = data
                for label in self.train_data[ind][1][s]:
                    note = label.data[1]
                    output[j, self.note_to_class(note)] = 1
            yield self.aggregate_raw_batch(features[:, :, None], output)

    def eval_set(self, set_name):
        if not self._eval_sets:
            for name in ['valid', 'test']:
                data = self.valid_data if name == 'valid' else self.test_data
                x, y = self.construct_eval_set(data)
                x, y = self.aggregate_raw_batch(x[:, :, None], y)
                self._eval_sets[name] = (x, y)
        return self._eval_sets[set_name]
