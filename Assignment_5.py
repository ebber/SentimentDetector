import math, os, pickle, re
from typing import Tuple, List, Dict


class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        pickled version of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.pos_filename: str = "pos.dat"
        self.neg_filename: str = "neg.dat"
        self.training_data_directory: str = "movie_reviews/"
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
            print("Data files found - loading to use pickled values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
        else:
            print("Data files not found - running training...")
            self.train()

    def train(self) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates 'self.pos_freqs' and 'self.neg_freqs' dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
        # Get the list of file names from the training data directory
        # os.walk returns a generator (feel free to Google "python generators" if you're
        # curious to learn more, next gets the first value from this generator) that gives
        # triples of (current_path, sub_directories, files). We want the "files" so we access
        # the 3rd part of the triple
        files: List[str] = next(os.walk(self.training_data_directory))[2]

        # files now holds a list of the filenames
        # self.training_data_directory holds the folder name where these files are

        # *Tip:* training can take a while, to make it more transparent, we can use the
        # enumerate function, which loops over something and has an automatic counter.
        # write something like this to track progress
        # for index, filename in enumerate(files, 1):
        #     print(f"Training on file {index} of {len(files)}")
        #     <the rest of your code for updating frequencies here>

        # we want to fill pos_freqs and neg_freqs with the correct counts of words from
        # their respective reviews

        # below is how you would load a file with filename given by `f_name`
        # `text` here will be the literal text of the file (i.e. what you would see
        # if you opened the file in a text editor
        # text = self.load_file(os.path.join(self.training_data_directory, f_name))

        # for each file, if it is a negative file, update the frequencies in the negative
        # frequency dictionary. If it is a positive file, update the frequencies in the
        # positive frequency dictionary. If it is neither a positive or negative file,
        # ignore it and move to the next file

        # Updating frequences: to update the frequencies for each file, you need to get
        # the text of the file, tokenize it, then update the appropriate dictionary for
        # those tokens. We've asked you to write a function `update_dict` that will make
        # your life easier here. Write that function first then pass it your list of
        # tokens from the file and the appropriate dictionary
        
        # for debugging purposes, it might be useful to print out the tokens and their
        # frequencies for both the positive and negative dictionaries

        # once you have gone through all the files, save the frequency dictionaries to
        # avoid extra work in the future (using the save_dict method). The objects you
        # are saving are self.pos_freqs and self.neg_freqs and the filepaths to save to
        # are self.pos_filename and self.neg_filename

    def classify(self, text: str) -> str:
        """Classifies given text as positive or negative from calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification, either positive or negative
        """

        pass
        
        # get a list of the individual tokens that occur in text
        
        # create some variables to store the positive and negative probabilities. since
        # we will be adding logs of probabilities, the initial values for the positive
        # and negative probabilities are set to 0

        # get the sum of all of the frequencies of the features in each document class
        # (i.e. how many words occurred in total across all documents for the given class) - 
        # will be used in calculating the probability of each document class given each
        # individual feature

        # for each token in the text, calculate the probability of it occurring in a
        # positive document and in a negative document and add the logs of those to the
        # running sums. when calculating the probabilities, always add 1 to the numerator
        # of each probability for add one smoothing (so that we never have a probability
        # of 0)

        # for debugging purposes, it may help to print the overall positive and negative
        # probabilities

        # determine whether positive or negative was more probable (i.e. which one was
        # larger)

        # return a string of "positive" or "negative"

    def load_file(self, filepath: str) -> str:
        """Loads text of given file

        Args:
            filepath - relative path to file to load

        Returns:
            text of the given file
        """
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name

        Args:
            dict - a dictionary to pickle
            filepath - relative path to file to save
        """
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file

        Args:
            filepath - relative path to file to load

        Returns:
            dictionary stored in given file
        """
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        """Splits given text into a list of the individual tokens in order

        Args:
            text - text to tokenize

        Returns:
            tokens of given text in order
        """
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        """Updates given (word -> frequency) dictionary with given words list

        By updating we mean increment the count of each word in words in the dictionary.
        If any word in words is not currently in the dictionary add it with a count of 1.
        (if a word is in words multiple times you'll increment it as many times
        as it appears)

        Args:
            words - list of tokens to update frequencies of
            freqs - dictionary of frequencies to update
        """
        pass



if __name__ == "__main__":
    b = BayesClassifier()
    a_list_of_words = ["I", "really", "like", "this", "movie", ".", "I", "hope", \
                       "you", "like", "it", "too"]
    a_dictionary = {}
    b.update_dict(a_list_of_words, a_dictionary)
    assert a_dictionary["I"] == 2, "update_dict test 1"
    assert a_dictionary["like"] == 2, "update_dict test 2"
    assert a_dictionary["really"] == 1, "update_dict test 3"
    assert a_dictionary["too"] == 1, "update_dict test 4"

    assert sum(b.pos_freqs.values()) == 612445, "pos denominator test"
    assert sum(b.neg_freqs.values()) == 129404, "neg denominator test"

    assert b.pos_freqs['love'] == 1526, "word test 1"
    assert b.neg_freqs['love'] == 99, "word test 2"
    assert b.pos_freqs['terrible'] == 37, "word test 3"
    assert b.neg_freqs['terrible'] == 176, "word test 4"
    assert b.pos_freqs['the'] == 30590, "word test 5"
    assert b.neg_freqs['the'] == 5646, "word test 6"
    assert b.pos_freqs['computer'] == 46, "word test 7"
    assert b.neg_freqs['computer'] == 12, "word test 8"

    assert b.classify('I love computer science') == "positive", "classify test 1"
    assert b.classify('this movie is fantastic') == "positive", "classify test 2"
    assert b.classify('great') == "positive", "classify test 3"
    assert b.classify('rainy days are the worst') == "negative", "classify test 4"
    assert b.classify('computer science is terrible') == "negative", "classify test 5"

    #assuming that this token doesn't occur in either corpus, should default to negative
    #    because smaller denominator
    assert b.classify('blaaaaaaa') == "negative", "classify test 6"

    print("All tests passed!")
