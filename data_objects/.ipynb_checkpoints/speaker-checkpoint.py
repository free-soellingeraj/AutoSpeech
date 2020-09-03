from data_objects.utterance import Utterance
from pathlib import Path


# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, sources):
        self.root = root
        # self.partition = partition
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        self.sources = sources
        # if self.partition is None:
        #     p = self.root.joinpath("_sources.txt")
        # else:
        #     p = self.root.joinpath("_sources_{}.txt".format(self.partition))
        # 
        # with open(p, "r") as sources_file:
        #     sources = [
        #         l.strip().split(",") for l in sources_file
        #     ]
        # self.sources = [
        #     [self.root, frames_fname, self.name, wav_path] 
        #     for frames_fname, wav_path 
        #     in sources
        # ]

    def _load_utterances(self):
        self.utterances = [
            Utterance(source['root'].joinpath(source['frames_fname'])) 
            for source in self.sources
        ]

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a