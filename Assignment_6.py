import math, os, pickle, re
from typing import Tuple, List, Dict
import random

logging_level = 1

class BayesClassifier:
    """A Naive Bayes Classifier.
    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
        n - total number of data files
        pos_n - total number of positive data files
        neg_n - total number of negative data files
        k - for k-fold cross validation
        sets - a list of lists - the k sets of file names
    """

    def __init__(self, training_dir: str = "movie_reviews/", k_folds: int = 10):
        """Initialize BayesClassifier with optional configuration.
        
        Args:
            training_dir: Path to training data directory
            k_folds: Number of folds for k-fold cross validation
        
        Raises:
            ValueError: If training_dir doesn't exist or k_folds <= 0
        """
        # Validate training directory exists
        if not os.path.exists(training_dir):
            raise ValueError(f"Training directory not found: {training_dir}")
        
        # Validate k parameter
        if k_folds <= 0:
            raise ValueError(f"k must be positive, got: {k_folds}")
        
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.training_data_directory: str = training_dir
        self.neg_file_prefix: str = "movies-1"
        self.pos_file_prefix: str = "movies-5"
        #data members added for cross validation
        self.n: int = 0     #total number of files
        self.pos_n: int = 0 #total number of pos files
        self.neg_n: int = 0 #total number of neg files
        self.k: int = k_folds
        self.sets: List[List[str]] = [] #k sets of filenames for k-fold cross validation

    def train(self, files: List[str]) -> None:
        """Trains the Naive Bayes Classifier.
        Train here means generates 'self.pos_freqs' and 'self.neg_freqs' dictionaries with
        frequencies of words in corresponding positive/negative reviews. Additionally
        sets self.pos_n and self.neg_n with the appropriate values (number of positive
        files and negative files in files).

        Args: files - a list of files to use as training data.

        Returns: None
        
        Raises:
            ValueError: If files list is None or empty
        """
        # Validate input
        if files is None:
            raise ValueError("files list cannot be None")
        if not files:
            raise ValueError("files list cannot be empty")

        #reset the following 4 attributes to wipe out any prior training
        self.pos_freqs = {}
        self.neg_freqs = {}
        self.pos_n = 0
        self.neg_n = 0
        
        failed_files = []

        for index, filename in enumerate(files, 1):
            # Progress tracking for large datasets
            if logging_level >= 1 and index % 100 == 0:
                print(f"Training progress: {index}/{len(files)} files processed")
            
            try:
                text = self.load_file(os.path.join(self.training_data_directory, filename))
                tokens: List[str] = self.tokenize(text)

                if filename.startswith(self.pos_file_prefix):
                    self.update_dict(tokens, self.pos_freqs)
                    self.pos_n += 1
                elif filename.startswith(self.neg_file_prefix):
                    self.update_dict(tokens, self.neg_freqs)
                    self.neg_n += 1
            except Exception as e:
                # Handle file read errors gracefully - log and continue
                failed_files.append((filename, str(e)))
                if logging_level >= 1:
                    print(f"Warning: Failed to process file '{filename}': {e}")
        
        # Report failed files at end
        if failed_files and logging_level >= 1:
            print(f"Training completed with {len(failed_files)} failed files out of {len(files)}")
        
        # Ensure we trained on at least some files
        if self.pos_n == 0 and self.neg_n == 0:
            raise ValueError("Training failed: No valid files were processed")

    def classify(self, text: str) -> str:
        """Classifies given text as positive or negative by calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification as a str, either positive or negative
            
        Raises:
            ValueError: If text is None/empty or classifier not trained
        """
        # Validate input
        if text is None:
            raise ValueError("text cannot be None")
        if not text.strip():
            raise ValueError("text cannot be empty")
        
        # Check classifier is trained
        if not self.pos_freqs and not self.neg_freqs:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Prevent division by zero
        if self.pos_n + self.neg_n == 0:
            raise ValueError("No training data: pos_n and neg_n are both 0")
        
        tokens = self.tokenize(text)

        #initialize the probabilities with the prior probabilities of each class
        pos_prob = math.log(self.pos_n/(self.pos_n+self.neg_n))
        neg_prob = math.log(self.neg_n/(self.pos_n+self.neg_n))

        num_pos_words = sum(self.pos_freqs.values())
        num_neg_words = sum(self.neg_freqs.values())

        for word in tokens:
            num_pos_appearances = 1
            if word in self.pos_freqs:
                num_pos_appearances += self.pos_freqs[word]

            pos_prob += math.log(num_pos_appearances / num_pos_words)

            num_neg_appearances = 1
            if word in self.neg_freqs:
                num_neg_appearances += self.neg_freqs[word]

            neg_prob += math.log(num_neg_appearances / num_neg_words)

        if pos_prob > neg_prob:
            return "positive"
        else:
            return "negative"

    def load_file(self, filepath: str) -> str:
        """Loads text of given file

        Args:
            filepath - relative path to file to load

        Returns:
            text of the given file
            
        Raises:
            ValueError: If filepath is None or empty
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is incompatible
        """
        # Validate input
        if filepath is None:
            raise ValueError("filepath cannot be None")
        if not filepath.strip():
            raise ValueError("filepath cannot be empty")
        
        # Handle file operations with explicit error handling
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            with open(filepath, "r", encoding='utf8') as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding, e.object, e.start, e.end,
                f"Failed to decode file '{filepath}': {e.reason}"
            )

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name

        Args:
            dict - a dictionary to pickle
            filepath - relative path to file to save
            
        Raises:
            ValueError: If filepath is None/empty
            PermissionError: If no write permission
            IOError: If disk is full or other I/O error
        """
        # Validate input
        if filepath is None:
            raise ValueError("filepath cannot be None")
        if not filepath.strip():
            raise ValueError("filepath cannot be empty")
        
        try:
            print(f"Dictionary saved to file: {filepath}")
            with open(filepath, "wb") as f:
                pickle.Pickler(f).dump(dict)
        except PermissionError as e:
            raise PermissionError(f"No write permission for file: {filepath}") from e
        except IOError as e:
            raise IOError(f"I/O error writing to {filepath}: {e}") from e

    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file

        Args:
            filepath - relative path to file to load

        Returns:
            dictionary stored in given file
            
        Raises:
            ValueError: If filepath is None/empty
            FileNotFoundError: If file doesn't exist
            pickle.UnpicklingError: If file is corrupted
        """
        # Validate input
        if filepath is None:
            raise ValueError("filepath cannot be None")
        if not filepath.strip():
            raise ValueError("filepath cannot be empty")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dictionary file not found: {filepath}")
        
        try:
            print(f"Loading dictionary from file: {filepath}")
            with open(filepath, "rb") as f:
                return pickle.Unpickler(f).load()
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(f"Corrupted pickle file '{filepath}': {e}")

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
        for word in words:
            if word in freqs:
                freqs[word] += 1
            else:
                freqs[word] = 1

    def split(self) -> None:
        """ Splits the files into k sets. Positive files and negative files must
        be spread evenly among the sets. That is, separate the files into a
        list of positive files and a list of negative files. Divide each of those
        lists into k equal sizes chunks and store them in self.sets (the list of k
        sets). Note that it might not be possible for the files to be divided
        evenly into k sets, the size of the sets might be off by one.

        Returns: None
        
        Raises:
            ValueError: If training directory doesn't exist, is empty, or has fewer files than k
        """
        # Validate directory exists
        if not os.path.exists(self.training_data_directory):
            raise ValueError(f"Training directory not found: {self.training_data_directory}")
        
        # Validate k is positive
        if self.k <= 0:
            raise ValueError(f"k must be positive, got: {self.k}")
        
        #reset the sets
        self.sets = []
        
        # Get files and validate
        try:
            files: List[str] = next(os.walk(self.training_data_directory))[2]
        except StopIteration:
            raise ValueError(f"Cannot access directory: {self.training_data_directory}")
        
        if not files:
            raise ValueError(f"Training directory is empty: {self.training_data_directory}")
        
        if len(files) < self.k:
            raise ValueError(f"Not enough files ({len(files)}) for {self.k}-fold cross-validation")
        
        random.shuffle(files)
        increment= len(files) // self.k
        for i in range(self.k):
            self.sets.append(files[i*increment:(i+1)*increment])
        

    def classify_all(self, testing_data_set: List[str]) -> List[Tuple[str, str, str]]:
        """Runs self.classify on the contents of each file in the input list.

        Args:
            testing_data_set: a list of file names

        Returns:
            Produces and returns a list of tuples of the form....
            (file_name, truth value, classifier result)
            For example,
            [('movies-5-14993.txt', 'positive', 'positive'),
             ('movies-5-13188.txt', 'positive', 'positive'),
             ('movies-5-7898.txt', 'positive', 'negative'),
            ...]
            
        Raises:
            ValueError: If testing_data_set is None or empty
        """
        # Validate input
        if testing_data_set is None:
            raise ValueError("testing_data_set cannot be None")
        if not testing_data_set:
            raise ValueError("testing_data_set cannot be empty")
        
        classification_results = []
        failed_files = []
        
        for index, file in enumerate(testing_data_set, 1):
            # Progress indicator for large datasets
            if logging_level >= 1 and index % 50 == 0:
                print(f"Classification progress: {index}/{len(testing_data_set)} files")
            
            try:
                text = self.load_file(os.path.join(self.training_data_directory, file))
                classification = self.classify(text)
                truth_value = "positive" if file.startswith(self.pos_file_prefix) else "negative"
                classification_results.append((file, truth_value, classification))
            except Exception as e:
                # Handle file read failures gracefully - log and continue
                failed_files.append((file, str(e)))
                if logging_level >= 1:
                    print(f"Warning: Failed to classify file '{file}': {e}")
        
        # Report failed files at end
        if failed_files and logging_level >= 1:
            print(f"Classification completed with {len(failed_files)} failed files out of {len(testing_data_set)}")
        
        return classification_results

    def analyze_results(self, classy_results: List[Tuple[str, str, str]]) -> Tuple[float, float, float, float, float, float, float]:
        """Given a list of classification results as input, computes and returns
        a list of values for the performance metrics.

        Args:
            classy_results: classy_results will be a list of tuples of the following format....
            (file_name, truth value, classifier result)
            For example,
            [('movies-5-14993.txt', 'positive', 'positive'),
             ('movies-5-13188.txt', 'positive', 'positive'),
             ('movies-5-7898.txt', 'positive', 'negative'),
            ...]

        Returns:
            Using the classy_results data, this function will produce and return a
            tuple of the following metrics:
            (accuracy, pos_precision, pos_recall, pos_f1, neg_precision, neg_recall, neg_f1)
            
        Raises:
            ValueError: If classy_results is None or empty
        """
        # Validate input
        if classy_results is None:
            raise ValueError("classy_results cannot be None")
        if not classy_results:
            raise ValueError("classy_results cannot be empty")

        #Define TP, FP, TN, FN
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for result in classy_results:
            if result[1] == "positive" and result[2] == "positive":
                TP += 1
            elif result[1] == "positive" and result[2] == "negative":
                FN += 1
            elif result[1] == "negative" and result[2] == "positive":
                FP += 1
            elif result[1] == "negative" and result[2] == "negative":
                TN += 1

        # Calculate total for accuracy
        total = TP + TN + FP + FN
        if total == 0:
            raise ValueError("No valid classification results to analyze")

        #calculate metrics with division by zero protection
        accuracy = (TP + TN) / total
        
        # Positive metrics with division by zero handling
        pos_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        pos_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall) 
                  if (pos_precision + pos_recall) > 0 else 0.0)
        
        # Negative metrics with division by zero handling
        neg_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        neg_recall = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)
                  if (neg_precision + neg_recall) > 0 else 0.0)

        return (accuracy, pos_precision, pos_recall, pos_f1, neg_precision, neg_recall, neg_f1)
    def calculate_averages(self, k_sets_of_metrics: List[Tuple]) -> List[float]:
        """Calculates and returns the average of each of the metrics across the k runs.

        Args:
            k_sets_of_metrics:
            a list of k tuples (each of 7 items - performance metric data). For example,
            [(0.8057761732851986, 0.9720044792833147, 0.7805755395683454, 0.8658354114713217, 0.5040650406504065, 0.9084249084249084, 0.6483660130718955),
             (0.8067772170151406, 0.9806598407281001, 0.7744833782569631, 0.8654618473895581, 0.5059055118110236, 0.9379562043795621, 0.6572890025575447),
             (0.7936507936507936, 0.9704209328782708, 0.766397124887691, 0.856425702811245, 0.48717948717948717, 0.9047619047619048, 0.6333333333333333),
             ...]

        Returns:
            Produces and returns a list of 7 items, the average value for each metric across the k runs.
            
        Raises:
            ValueError: If k_sets_of_metrics is None, empty, or tuples have inconsistent sizes
        """
        # Validate input
        if k_sets_of_metrics is None:
            raise ValueError("k_sets_of_metrics cannot be None")
        if not k_sets_of_metrics:
            raise ValueError("k_sets_of_metrics cannot be empty")
        
        # Validate all tuples have 7 elements
        for i, metrics_tuple in enumerate(k_sets_of_metrics):
            if len(metrics_tuple) != 7:
                raise ValueError(f"Tuple at index {i} has {len(metrics_tuple)} elements, expected 7")
        
        #tuple lookst like (accuracy, pos_precision, pos_recall, pos_f1, neg_precision, neg_recall, neg_f1)
        num_tuples = len(k_sets_of_metrics)
        accuracy = 0
        pos_precision = 0
        pos_recall = 0
        pos_f1 = 0
        neg_precision = 0
        neg_recall = 0
        neg_f1 = 0

        for tuple in k_sets_of_metrics:
            accuracy += tuple[0]
            pos_precision += tuple[1]
            pos_recall += tuple[2]
            pos_f1 += tuple[3]
            neg_precision += tuple[4]
            neg_recall += tuple[5]
            neg_f1 += tuple[6]
        
        accuracy /= num_tuples
        pos_precision /= num_tuples
        pos_recall /= num_tuples
        pos_f1 /= num_tuples
        neg_precision /= num_tuples
        neg_recall /= num_tuples
        neg_f1 /= num_tuples

        return [accuracy, pos_precision, pos_recall, pos_f1, neg_precision, neg_recall, neg_f1]

    def evaluate(self) -> None:
        """ This method drives the k-fold cross validation process. First, it calls the
        split method to generate k sets of filenames, stored in self.sets. Next, it loops
        over those sets, letting each have a turn being the testing data (training a
        classifier with the other 9 sets).  More details can be found in the assignment pdf.

        Returns: None
        
        Raises:
            ValueError: If split fails or produces no valid sets
        """
        try:
            self.split() #split the data (file names) into self.k sets, stored self.sets
            
            # Validate split succeeded
            if not self.sets:
                raise ValueError("split() failed to create any sets")
            if len(self.sets) != self.k:
                raise ValueError(f"split() created {len(self.sets)} sets, expected {self.k}")
            
            k_metrics = []
            failed_folds = []
            
            for i in range(self.k): #execute k-fold cross validation
                try:
                    if logging_level >= 1:
                        print(f"\nProcessing fold {i+1}/{self.k}")
                    
                    td = self.sets[0:i] + self.sets[i+1:] #grab everything other than set i
                    training_data = []
                    for lst in td:
                        training_data += lst
                    self.train(training_data) #training on all sets other than set i

                    testing_data = self.sets[i] #testing data is set i
                    classification_results = self.classify_all(testing_data)
                    metrics = self.analyze_results(classification_results)
                    k_metrics.append(metrics)
                except Exception as e:
                    # Handle errors in individual fold processing gracefully
                    failed_folds.append((i, str(e)))
                    if logging_level >= 1:
                        print(f"Error processing fold {i+1}: {e}")
            
            # Check if we have any successful folds
            if not k_metrics:
                raise ValueError(f"All {self.k} folds failed to process. Check errors above.")
            
            # Report failed folds
            if failed_folds and logging_level >= 1:
                print(f"\nWarning: {len(failed_folds)} out of {self.k} folds failed")
            
            summary_results = self.calculate_averages(k_metrics)

            print(f"\n{'='*60}")
            print(f"K-Fold Cross-Validation Results ({len(k_metrics)}/{self.k} successful folds)")
            print(f"{'='*60}")
            print(f"average accuracy:      {summary_results[0]:.4f}")
            print(f"positive precision:    {summary_results[1]:.4f}")
            print(f"positive recall:       {summary_results[2]:.4f}")
            print(f"positive f-measure:    {summary_results[3]:.4f}")
            print(f"negative precision:    {summary_results[4]:.4f}")
            print(f"negative recall:       {summary_results[5]:.4f}")
            print(f"negative f-measure:    {summary_results[6]:.4f}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\nEvaluation failed: {e}")
            raise

if __name__ == "__main__":
    b = BayesClassifier()
    b.evaluate()

    b.split()
    first_file_1 = b.sets[0][0]

    assert len(b.sets) == b.k, "split test 1, tests that there are the right number of sets"
    assert abs(len(b.sets[0]) - len(b.sets[1])) < 3, "split test 2, tests that the first set is not too different from the second set"
    assert abs(len(b.sets[0]) - len(b.sets[-1])) < 3, "split test 3, tests that the first set is not too different from the last set"
    assert first_file_1 not in b.sets[1], "split test 4, tests that the first file is not in the second set"

    b.split()
    first_file_2 = b.sets[0][0]
    assert first_file_2 not in b.sets[1], "split test 5, tests that the first file is not in the second set"

    first_file_2 = b.sets[0][0]
    #the following test makes sure that split is shuffling the data
    assert first_file_1 != first_file_2, "split test 6, tests that we shuffle every split"

    print("All split tests passed!")

    some_files = ['movies-5-2997.txt', 'movies-5-14493.txt', 'movies-5-5803.txt',
                  'movies-5-15857.txt', 'movies-5-12100.txt', 'movies-5-13604.txt',
                  'movies-5-12831.txt', 'movies-5-412.txt', 'movies-5-21825.txt',
                  'movies-5-23469.txt', 'movies-5-3089.txt', 'movies-5-13672.txt',
                  'movies-1-13558.txt', 'movies-1-20209.txt', 'movies-1-13936.txt',
                  'movies-1-18942.txt', 'movies-1-3489.txt', 'movies-1-19588.txt',
                  'movies-1-4738.txt', 'movies-1-20549.txt']

    files: List[str] = next(os.walk(b.training_data_directory))[2]
    b.train(files)
    results = b.classify_all(some_files)

    #tests that classify_all returns a list of the same length
    assert len(results) == len(some_files), "classify_all test 1"

    #tests that each item in the returned list is a list or tuple of 3 items
    assert len(results[0]) == 3, "classify_all test 2"

    #tests the truth data for a positive file
    assert results[0][1] == "positive", "classify_all test 3"

    #tests the truth data for a negative file
    assert results[14][1] == "negative", "classify_all test 4"
    print("All classify_all tests passed!")

    classy = [('movies-5-2997.txt', 'positive', 'positive'),
              ('movies-5-14493.txt', 'positive', 'positive'),
              ('movies-5-5803.txt', 'positive', 'positive'),
              ('movies-5-15857.txt', 'positive', 'negative'),
              ('movies-5-12100.txt', 'positive', 'positive'),
              ('movies-5-13604.txt', 'positive', 'positive'),
              ('movies-5-12831.txt', 'positive', 'positive'),
              ('movies-5-412.txt', 'positive', 'positive'),
              ('movies-5-21825.txt', 'positive', 'positive'),
              ('movies-5-23469.txt', 'positive', 'positive'),
              ('movies-5-12331.txt', 'positive', 'negative'),
              ('movies-5-3089.txt', 'positive', 'positive'),
              ('movies-5-13672.txt', 'positive', 'positive'),
              ('movies-1-13558.txt', 'negative', 'negative'),
              ('movies-1-20209.txt', 'negative', 'negative'),
              ('movies-1-13936.txt', 'negative', 'negative'),
              ('movies-1-18942.txt', 'negative', 'negative'),
              ('movies-1-3489.txt', 'negative', 'negative'),
              ('movies-1-19588.txt', 'negative', 'negative'),
              ('movies-1-4738.txt', 'negative', 'negative'),
              ('movies-1-20549.txt', 'negative', 'positive')]

    metrics = b.analyze_results(classy)

    assert metrics[0] == 0.8571428571428571, "accuracy test"
    assert metrics[1] == 0.9166666666666666, "positive precision test"
    assert metrics[2] == 0.8461538461538461, "positive recall test"
    assert metrics[3] == 0.8799999999999999, "positive f1 test"
    assert metrics[4] == 0.7777777777777778, "negative precision test"
    assert metrics[5] == 0.875, "negative recall test"
    assert metrics[6] == 0.823529411764706, "negative f1 test"

    print("All analyze_results tests passed!")

    k_results = [(0.8115523465703971, 0.9829738933030647, 0.7787769784172662, 0.8690416457601605,
                  0.5119047619047619, 0.945054945054945, 0.664092664092664),
                 (0.8168709444844989, 0.97669256381798, 0.7906558849955077, 0.8738828202581926,
                  0.5205761316872428, 0.9233576642335767, 0.6657894736842106),
                 (0.8037518037518038, 0.9827784156142365, 0.7690925426774483, 0.8629032258064515,
                  0.5009708737864078, 0.945054945054945, 0.6548223350253807),
                 (0.7952415284787311, 0.9769850402761795, 0.7628032345013477, 0.8567103935418768,
                  0.49034749034749037, 0.927007299270073, 0.6414141414141414),
                 (0.8167388167388168, 0.97669256381798, 0.7906558849955077, 0.8738828202581926,
                  0.5195876288659794, 0.9230769230769231, 0.6649076517150396),
                 (0.8010093727469358, 0.9634551495016611, 0.7816711590296496, 0.8630952380952381,
                  0.49793388429752067, 0.8795620437956204, 0.6358839050131926),
                 (0.8080808080808081, 0.9721293199554069, 0.7834681042228212, 0.8676616915422886,
                  0.5071574642126789, 0.9084249084249084, 0.6509186351706037),
                 (0.802451333813987, 0.9718785151856018, 0.7762803234501348, 0.8631368631368631,
                  0.5, 0.9087591240875912, 0.6450777202072538),
                 (0.810966810966811, 0.9818799546998868, 0.7789757412398922, 0.8687374749498998,
                  0.510934393638171, 0.9413919413919414, 0.6623711340206185),
                 (0.8103821196827685, 0.9786036036036037, 0.7807726864330637, 0.8685657171414293,
                  0.5110220440881763, 0.9306569343065694, 0.6597671410090555)]
    # For each metric index, print whether all values in k_results for that metric are close to the corresponding summary_result
    summary_results = [0.8077045885315558, 0.9764069019775603, 0.779315253996264, 0.8667617890490593,
                       0.5070434672828428, 0.9232346728697094, 0.654504480135216]

    results = b.calculate_averages(k_results)

    if logging_level >= 1:
        print(list(map(lambda x: x[0] == x[1], zip(results, summary_results))))
    if logging_level >= 1:
        print(results)

    assert len(b.calculate_averages(k_results)) == 7, "calculate averages test 1"
    assert b.calculate_averages(k_results) == summary_results, "calculate averages test 2"

    b.k = 2
    k_results = [(0.8115523465703971, 0.9829738933030647, 0.7787769784172662, 0.8690416457601605,
                0.5119047619047619, 0.945054945054945, 0.664092664092664),
                (0.8168709444844989, 0.97669256381798, 0.7906558849955077, 0.8738828202581926,
                0.5205761316872428, 0.9233576642335767, 0.6657894736842106)]
    summary_results = [0.814211645527448, 0.9798332285605224, 0.7847164317063869, 0.8714622330091766,
                    0.5162404467960023, 0.9342063046442608, 0.6649410688884373]
    assert b.calculate_averages(k_results) == summary_results, "calculate averages test 3"
    print("All calculate averages tests passed!")
    print("All tests passed!!")

    
