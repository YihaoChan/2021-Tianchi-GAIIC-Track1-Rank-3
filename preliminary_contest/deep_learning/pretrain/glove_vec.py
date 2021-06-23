from glove import Corpus
from glove import Glove
import warnings
import directory
import multiprocessing

warnings.filterwarnings('ignore')


class GloveConfig(object):
    def __init__(self):
        self.window = 8

        self.no_components = 100

        self.learning_rate = 1e-3

        self.epochs = 50

        self.no_threads = multiprocessing.cpu_count()


class GloveVec(object):
    def __init__(self):
        glove_config = GloveConfig()

        self.window = glove_config.window

        self.no_components = glove_config.no_components

        self.learning_rate = glove_config.learning_rate

        self.epochs = glove_config.epochs

        self.no_threads = glove_config.no_threads

    def glove_pretrain(self):
        corpus = []

        with open(directory.CORPUS_PATH, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                corpus.append(line)

        corpus_model = Corpus()

        corpus_model.fit(corpus, window=self.window)

        glove_model = Glove(no_components=self.no_components, learning_rate=self.learning_rate, random_state=7)

        glove_model.fit(corpus_model.matrix, epochs=self.epochs, no_threads=self.no_threads, verbose=True)

        glove_model.add_dictionary(corpus_model.dictionary)

        glove_model.save(directory.GLOVE_VECTOR_PATH)
