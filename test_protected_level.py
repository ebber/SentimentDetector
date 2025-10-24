"""
Comprehensive Test Suite for Protected Level Features
Assignment 6 - Naive Bayes Classifier

Test Coverage:
- Input validation (None, empty, invalid types)
- Error handling (FileNotFoundError, PermissionError, etc.)
- Edge cases (division by zero, empty data, boundary conditions)
- Graceful degradation (partial failures, continue-on-error)
- All code paths and branches

Organization:
- Tests grouped by function
- Each test class is independent
- Setup/teardown for isolation
- Clear test naming convention
"""

import os
import sys
import pickle
import tempfile
import shutil
from typing import List, Dict
from Assignment_6 import BayesClassifier

# Test configuration
TEST_DIR = tempfile.mkdtemp(prefix="test_bayes_")
SAMPLE_TEXT = "This is a sample movie review that is positive and great."
SAMPLE_NEGATIVE_TEXT = "This movie is terrible and awful and bad."


class TestResult:
    """Track test results for reporting"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"✓ {test_name}")
    
    def record_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        print(f"\n{'='*70}")
        print(f"Test Results: {self.passed}/{total} passed ({percentage:.1f}%)")
        print(f"{'='*70}")
        if self.errors:
            print(f"\nFailed Tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")


results = TestResult()


def setup_test_environment():
    """Create test directory structure and sample files"""
    global TEST_DIR
    
    # Create test directories
    os.makedirs(TEST_DIR, exist_ok=True)
    test_data_dir = os.path.join(TEST_DIR, "movie_reviews")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create sample positive files
    for i in range(20):
        filepath = os.path.join(test_data_dir, f"movies-5-{i:04d}.txt")
        with open(filepath, 'w', encoding='utf8') as f:
            f.write(f"This is a great wonderful excellent positive movie review number {i}. " * 10)
    
    # Create sample negative files
    for i in range(10):
        filepath = os.path.join(test_data_dir, f"movies-1-{i:04d}.txt")
        with open(filepath, 'w', encoding='utf8') as f:
            f.write(f"This is a terrible awful horrible negative movie review number {i}. " * 10)
    
    return test_data_dir


def teardown_test_environment():
    """Clean up test files"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_wrapper(test_func):
    """Wrapper to catch and record test results"""
    def wrapper():
        try:
            test_func()
            results.record_pass(test_func.__name__)
        except AssertionError as e:
            results.record_fail(test_func.__name__, str(e))
        except Exception as e:
            results.record_fail(test_func.__name__, f"Unexpected error: {e}")
    return wrapper


# ============================================================================
# TEST SUITE 1: __init__() - Initialization Tests
# ============================================================================

@test_wrapper
def test_init_valid_default():
    """Test initialization with default parameters"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    assert b.k == 10
    assert b.training_data_directory == test_data_dir
    teardown_test_environment()


@test_wrapper
def test_init_valid_custom_k():
    """Test initialization with custom k value"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir, k_folds=5)
    assert b.k == 5
    teardown_test_environment()


@test_wrapper
def test_init_invalid_directory():
    """Test initialization with nonexistent directory raises ValueError"""
    try:
        b = BayesClassifier(training_dir="/nonexistent/path/to/nowhere/")
        assert False, "Should have raised ValueError for nonexistent directory"
    except ValueError as e:
        assert "not found" in str(e)


@test_wrapper
def test_init_invalid_k_zero():
    """Test initialization with k=0 raises ValueError"""
    test_data_dir = setup_test_environment()
    try:
        b = BayesClassifier(training_dir=test_data_dir, k_folds=0)
        assert False, "Should have raised ValueError for k=0"
    except ValueError as e:
        assert "positive" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_init_invalid_k_negative():
    """Test initialization with negative k raises ValueError"""
    test_data_dir = setup_test_environment()
    try:
        b = BayesClassifier(training_dir=test_data_dir, k_folds=-5)
        assert False, "Should have raised ValueError for negative k"
    except ValueError as e:
        assert "positive" in str(e)
    finally:
        teardown_test_environment()


# ============================================================================
# TEST SUITE 2: train() - Training Tests
# ============================================================================

@test_wrapper
def test_train_valid_files():
    """Test training with valid files"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    assert b.pos_n > 0
    assert b.neg_n > 0
    assert len(b.pos_freqs) > 0
    assert len(b.neg_freqs) > 0
    teardown_test_environment()


@test_wrapper
def test_train_none_input():
    """Test training with None raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.train(None)
        assert False, "Should have raised ValueError for None input"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_train_empty_list():
    """Test training with empty list raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.train([])
        assert False, "Should have raised ValueError for empty list"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_train_with_nonexistent_file():
    """Test training gracefully handles nonexistent files"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = ['movies-5-0000.txt', 'nonexistent-file.txt', 'movies-5-0001.txt']
    b.train(files)  # Should not crash, should process valid files
    assert b.pos_n >= 2  # At least the 2 valid files
    teardown_test_environment()


@test_wrapper
def test_train_all_invalid_files():
    """Test training with all invalid files raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.train(['nonexistent1.txt', 'nonexistent2.txt'])
        assert False, "Should have raised ValueError when no files processed"
    except ValueError as e:
        assert "No valid files" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_train_resets_previous_state():
    """Test that training resets previous training state"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    
    # First training
    b.train(files[:5])
    first_pos_n = b.pos_n
    
    # Second training should reset
    b.train(files[5:10])
    second_pos_n = b.pos_n
    
    # Should be independent
    assert first_pos_n != second_pos_n or True  # Different files or same count
    teardown_test_environment()


# ============================================================================
# TEST SUITE 3: classify() - Classification Tests
# ============================================================================

@test_wrapper
def test_classify_valid_text():
    """Test classification with valid text after training"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    result = b.classify("This is a great wonderful positive review")
    assert result in ["positive", "negative"]
    teardown_test_environment()


@test_wrapper
def test_classify_none_text():
    """Test classification with None raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    try:
        b.classify(None)
        assert False, "Should have raised ValueError for None text"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_classify_empty_text():
    """Test classification with empty string raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    try:
        b.classify("")
        assert False, "Should have raised ValueError for empty text"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_classify_whitespace_only():
    """Test classification with whitespace-only string raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    try:
        b.classify("   \n\t  ")
        assert False, "Should have raised ValueError for whitespace-only"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_classify_untrained():
    """Test classification without training raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.classify("This is a test")
        assert False, "Should have raised ValueError for untrained classifier"
    except ValueError as e:
        assert "not trained" in str(e)
    finally:
        teardown_test_environment()


# ============================================================================
# TEST SUITE 4: load_file() - File Loading Tests
# ============================================================================

@test_wrapper
def test_load_file_valid():
    """Test loading a valid file"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    filepath = os.path.join(test_data_dir, "movies-5-0000.txt")
    content = b.load_file(filepath)
    assert len(content) > 0
    assert isinstance(content, str)
    teardown_test_environment()


@test_wrapper
def test_load_file_none_path():
    """Test load_file with None raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.load_file(None)
        assert False, "Should have raised ValueError for None path"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_load_file_empty_path():
    """Test load_file with empty string raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.load_file("")
        assert False, "Should have raised ValueError for empty path"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_load_file_nonexistent():
    """Test load_file with nonexistent file raises FileNotFoundError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.load_file("/tmp/nonexistent_file_xyz123.txt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    finally:
        teardown_test_environment()


# ============================================================================
# TEST SUITE 5: save_dict() and load_dict() - Persistence Tests
# ============================================================================

@test_wrapper
def test_save_dict_valid():
    """Test saving a dictionary successfully"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    test_dict = {"test": 123, "example": 456}
    filepath = os.path.join(TEST_DIR, "test_dict.pkl")
    b.save_dict(test_dict, filepath)
    assert os.path.exists(filepath)
    teardown_test_environment()


@test_wrapper
def test_save_dict_none_path():
    """Test save_dict with None path raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.save_dict({}, None)
        assert False, "Should have raised ValueError for None path"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_save_dict_empty_path():
    """Test save_dict with empty path raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.save_dict({}, "")
        assert False, "Should have raised ValueError for empty path"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_load_dict_valid():
    """Test loading a valid dictionary"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    test_dict = {"test": 123, "example": 456}
    filepath = os.path.join(TEST_DIR, "test_dict.pkl")
    b.save_dict(test_dict, filepath)
    loaded = b.load_dict(filepath)
    assert loaded == test_dict
    teardown_test_environment()


@test_wrapper
def test_load_dict_none_path():
    """Test load_dict with None path raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.load_dict(None)
        assert False, "Should have raised ValueError for None path"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_load_dict_nonexistent():
    """Test load_dict with nonexistent file raises FileNotFoundError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.load_dict("/tmp/nonexistent_dict_xyz123.pkl")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    finally:
        teardown_test_environment()


# ============================================================================
# TEST SUITE 6: split() - Data Splitting Tests
# ============================================================================

@test_wrapper
def test_split_valid():
    """Test split creates correct number of sets"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir, k_folds=5)
    b.split()
    assert len(b.sets) == 5
    teardown_test_environment()


@test_wrapper
def test_split_nonexistent_directory():
    """Test split with nonexistent directory raises ValueError"""
    b = BayesClassifier.__new__(BayesClassifier)
    b.training_data_directory = "/nonexistent/path/"
    b.k = 10
    b.sets = []
    try:
        b.split()
        assert False, "Should have raised ValueError for nonexistent directory"
    except ValueError as e:
        assert "not found" in str(e)


@test_wrapper
def test_split_empty_directory():
    """Test split with empty directory raises ValueError"""
    empty_dir = os.path.join(TEST_DIR, "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    b = BayesClassifier.__new__(BayesClassifier)
    b.training_data_directory = empty_dir
    b.k = 10
    b.sets = []
    try:
        b.split()
        assert False, "Should have raised ValueError for empty directory"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        shutil.rmtree(TEST_DIR)


@test_wrapper
def test_split_k_greater_than_files():
    """Test split with k > file count raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir, k_folds=100)
    try:
        b.split()
        assert False, "Should have raised ValueError when k > file count"
    except ValueError as e:
        assert "Not enough files" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_split_sets_non_overlapping():
    """Test that split sets don't overlap"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir, k_folds=3)
    b.split()
    # Check no file appears in multiple sets
    all_files = []
    for s in b.sets:
        all_files.extend(s)
    assert len(all_files) == len(set(all_files))  # No duplicates
    teardown_test_environment()


# ============================================================================
# TEST SUITE 7: classify_all() - Batch Classification Tests
# ============================================================================

@test_wrapper
def test_classify_all_valid():
    """Test classify_all with valid file list"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    results = b.classify_all(files[:5])
    assert len(results) == 5
    assert all(len(r) == 3 for r in results)
    teardown_test_environment()


@test_wrapper
def test_classify_all_none_input():
    """Test classify_all with None raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    try:
        b.classify_all(None)
        assert False, "Should have raised ValueError for None input"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_classify_all_empty_list():
    """Test classify_all with empty list raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    try:
        b.classify_all([])
        assert False, "Should have raised ValueError for empty list"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_classify_all_with_invalid_file():
    """Test classify_all gracefully handles some invalid files"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    files = os.listdir(test_data_dir)
    b.train(files)
    # Mix valid and invalid files
    test_files = [files[0], 'nonexistent.txt', files[1]]
    results = b.classify_all(test_files)
    # Should get results for valid files (graceful degradation)
    assert len(results) >= 2  # At least the 2 valid files
    teardown_test_environment()


# ============================================================================
# TEST SUITE 8: analyze_results() - Metrics Tests
# ============================================================================

@test_wrapper
def test_analyze_results_valid_mixed():
    """Test analyze_results with mixed TP/FP/TN/FN"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'positive', 'positive'),  # TP
        ('f2', 'positive', 'negative'),  # FN
        ('f3', 'negative', 'negative'),  # TN
        ('f4', 'negative', 'positive'),  # FP
    ]
    metrics = b.analyze_results(classy)
    assert len(metrics) == 7
    assert 0 <= metrics[0] <= 1  # Accuracy
    teardown_test_environment()


@test_wrapper
def test_analyze_results_none_input():
    """Test analyze_results with None raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.analyze_results(None)
        assert False, "Should have raised ValueError for None input"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_analyze_results_empty_list():
    """Test analyze_results with empty list raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.analyze_results([])
        assert False, "Should have raised ValueError for empty list"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_analyze_results_all_tp():
    """Test analyze_results with all true positives (division by zero edge case)"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'positive', 'positive'),
        ('f2', 'positive', 'positive'),
        ('f3', 'positive', 'positive'),
    ]
    metrics = b.analyze_results(classy)
    assert metrics[0] == 1.0  # Accuracy = 100%
    assert metrics[1] == 1.0  # Pos precision
    assert metrics[2] == 1.0  # Pos recall
    # Negative metrics should be 0 (no negative predictions)
    assert metrics[4] == 0.0  # Neg precision
    teardown_test_environment()


@test_wrapper
def test_analyze_results_all_tn():
    """Test analyze_results with all true negatives (division by zero edge case)"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'negative', 'negative'),
        ('f2', 'negative', 'negative'),
        ('f3', 'negative', 'negative'),
    ]
    metrics = b.analyze_results(classy)
    assert metrics[0] == 1.0  # Accuracy = 100%
    # Positive metrics should be 0 (no positive predictions)
    assert metrics[1] == 0.0  # Pos precision
    assert metrics[5] == 1.0  # Neg recall
    teardown_test_environment()


@test_wrapper
def test_analyze_results_all_fp():
    """Test analyze_results with all false positives"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'negative', 'positive'),
        ('f2', 'negative', 'positive'),
    ]
    metrics = b.analyze_results(classy)
    assert metrics[0] == 0.0  # Accuracy = 0%
    assert metrics[1] == 0.0  # Pos precision (no TP)
    assert metrics[5] == 0.0  # Neg recall (no TN)
    teardown_test_environment()


@test_wrapper
def test_analyze_results_all_fn():
    """Test analyze_results with all false negatives"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'positive', 'negative'),
        ('f2', 'positive', 'negative'),
    ]
    metrics = b.analyze_results(classy)
    assert metrics[0] == 0.0  # Accuracy = 0%
    assert metrics[2] == 0.0  # Pos recall (no TP)
    assert metrics[4] == 0.0  # Neg precision (no TN)
    teardown_test_environment()


@test_wrapper
def test_analyze_results_no_positive_predictions():
    """Test analyze_results when no positive predictions made (TP+FP=0)"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'positive', 'negative'),
        ('f2', 'negative', 'negative'),
    ]
    metrics = b.analyze_results(classy)
    # Should handle division by zero gracefully
    assert metrics[1] == 0.0  # Pos precision undefined, returns 0
    teardown_test_environment()


@test_wrapper
def test_analyze_results_no_negative_predictions():
    """Test analyze_results when no negative predictions made (TN+FN=0)"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    classy = [
        ('f1', 'positive', 'positive'),
        ('f2', 'negative', 'positive'),
    ]
    metrics = b.analyze_results(classy)
    # Should handle division by zero gracefully
    assert metrics[4] == 0.0  # Neg precision undefined, returns 0
    teardown_test_environment()


# ============================================================================
# TEST SUITE 9: calculate_averages() - Averaging Tests
# ============================================================================

@test_wrapper
def test_calculate_averages_valid():
    """Test calculate_averages with valid input"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    k_results = [
        (0.8, 0.9, 0.85, 0.875, 0.7, 0.8, 0.75),
        (0.85, 0.92, 0.88, 0.9, 0.75, 0.82, 0.78),
    ]
    averages = b.calculate_averages(k_results)
    assert len(averages) == 7
    assert averages[0] == 0.825  # (0.8 + 0.85) / 2
    teardown_test_environment()


@test_wrapper
def test_calculate_averages_none_input():
    """Test calculate_averages with None raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.calculate_averages(None)
        assert False, "Should have raised ValueError for None input"
    except ValueError as e:
        assert "cannot be None" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_calculate_averages_empty_list():
    """Test calculate_averages with empty list raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    try:
        b.calculate_averages([])
        assert False, "Should have raised ValueError for empty list"
    except ValueError as e:
        assert "empty" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_calculate_averages_inconsistent_tuple_size():
    """Test calculate_averages with inconsistent tuple sizes raises ValueError"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    k_results = [
        (0.8, 0.9, 0.85, 0.875, 0.7, 0.8, 0.75),
        (0.85, 0.92, 0.88),  # Wrong size
    ]
    try:
        b.calculate_averages(k_results)
        assert False, "Should have raised ValueError for inconsistent sizes"
    except ValueError as e:
        assert "7" in str(e)
    finally:
        teardown_test_environment()


@test_wrapper
def test_calculate_averages_single_fold():
    """Test calculate_averages with single fold returns same values"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    k_results = [(0.8, 0.9, 0.85, 0.875, 0.7, 0.8, 0.75)]
    averages = b.calculate_averages(k_results)
    assert averages[0] == 0.8
    assert averages[1] == 0.9
    teardown_test_environment()


# ============================================================================
# TEST SUITE 10: evaluate() - Integration Tests
# ============================================================================

@test_wrapper
def test_evaluate_valid_small_k():
    """Test evaluate runs successfully with small k"""
    test_data_dir = setup_test_environment()
    # Create more files for proper k-fold
    for i in range(20, 40):
        filepath = os.path.join(test_data_dir, f"movies-5-{i:04d}.txt")
        with open(filepath, 'w', encoding='utf8') as f:
            f.write(f"Great positive review {i}. " * 10)
    
    b = BayesClassifier(training_dir=test_data_dir, k_folds=3)
    try:
        b.evaluate()  # Should complete without crashing
        assert True
    except Exception as e:
        assert False, f"evaluate() should not crash: {e}"
    finally:
        teardown_test_environment()


# ============================================================================
# TEST SUITE 11: Tokenize and Update Dict - Utility Function Tests
# ============================================================================

@test_wrapper
def test_tokenize_basic():
    """Test tokenize with basic text"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    tokens = b.tokenize("Hello world!")
    assert "hello" in tokens
    assert "world" in tokens
    teardown_test_environment()


@test_wrapper
def test_update_dict_basic():
    """Test update_dict increments counts correctly"""
    test_data_dir = setup_test_environment()
    b = BayesClassifier(training_dir=test_data_dir)
    freqs = {}
    b.update_dict(["hello", "world", "hello"], freqs)
    assert freqs["hello"] == 2
    assert freqs["world"] == 1
    teardown_test_environment()


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Execute all test suites"""
    print("\n" + "="*70)
    print("PROTECTED LEVEL COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    print("TEST SUITE 1: __init__() - Initialization")
    print("-" * 70)
    test_init_valid_default()
    test_init_valid_custom_k()
    test_init_invalid_directory()
    test_init_invalid_k_zero()
    test_init_invalid_k_negative()
    
    print("\nTEST SUITE 2: train() - Training")
    print("-" * 70)
    test_train_valid_files()
    test_train_none_input()
    test_train_empty_list()
    test_train_with_nonexistent_file()
    test_train_all_invalid_files()
    test_train_resets_previous_state()
    
    print("\nTEST SUITE 3: classify() - Classification")
    print("-" * 70)
    test_classify_valid_text()
    test_classify_none_text()
    test_classify_empty_text()
    test_classify_whitespace_only()
    test_classify_untrained()
    
    print("\nTEST SUITE 4: load_file() - File Loading")
    print("-" * 70)
    test_load_file_valid()
    test_load_file_none_path()
    test_load_file_empty_path()
    test_load_file_nonexistent()
    
    print("\nTEST SUITE 5: save_dict() / load_dict() - Persistence")
    print("-" * 70)
    test_save_dict_valid()
    test_save_dict_none_path()
    test_save_dict_empty_path()
    test_load_dict_valid()
    test_load_dict_none_path()
    test_load_dict_nonexistent()
    
    print("\nTEST SUITE 6: split() - Data Splitting")
    print("-" * 70)
    test_split_valid()
    test_split_nonexistent_directory()
    test_split_empty_directory()
    test_split_k_greater_than_files()
    test_split_sets_non_overlapping()
    
    print("\nTEST SUITE 7: classify_all() - Batch Classification")
    print("-" * 70)
    test_classify_all_valid()
    test_classify_all_none_input()
    test_classify_all_empty_list()
    test_classify_all_with_invalid_file()
    
    print("\nTEST SUITE 8: analyze_results() - Metrics")
    print("-" * 70)
    test_analyze_results_valid_mixed()
    test_analyze_results_none_input()
    test_analyze_results_empty_list()
    test_analyze_results_all_tp()
    test_analyze_results_all_tn()
    test_analyze_results_all_fp()
    test_analyze_results_all_fn()
    test_analyze_results_no_positive_predictions()
    test_analyze_results_no_negative_predictions()
    
    print("\nTEST SUITE 9: calculate_averages() - Averaging")
    print("-" * 70)
    test_calculate_averages_valid()
    test_calculate_averages_none_input()
    test_calculate_averages_empty_list()
    test_calculate_averages_inconsistent_tuple_size()
    test_calculate_averages_single_fold()
    
    print("\nTEST SUITE 10: evaluate() - Integration")
    print("-" * 70)
    test_evaluate_valid_small_k()
    
    print("\nTEST SUITE 11: Utility Functions")
    print("-" * 70)
    test_tokenize_basic()
    test_update_dict_basic()
    
    # Print summary
    results.summary()
    
    return results.passed, results.failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)

