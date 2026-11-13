from src.loaders import cifar10_loader as cf10
from src.loaders import cifar100_loader as cf100
from src.loaders.class_names import CLASS_NAMES
from src import formatter
from src import preprocessor
from src import splitter

def main():
    num_classes = len(CLASS_NAMES)

    cf10_train = cf10.generate_set()
    cf100_train = cf100.generate_set()

    merged_training_set = formatter.format_sets(cf10_train, cf100_train)

    final_training_set = preprocessor.apply_preprocessing(merged_training_set)

    cf10_test = cf10.generate_test_set()
    cf100_test = cf100.generate_test_set()

    merged_test_set = formatter.format_sets(cf10_test, cf100_test)
    final_test_set = preprocessor.apply_preprocessing(merged_test_set)

    X_train, y_train, X_val, y_val = splitter.split_validation_set(final_training_set)
    X_test, y_test = final_test_set['data'], final_test_set['labels']
    y_train, y_val, y_test = formatter.one_hot_encode(num_classes, y_train, y_val, y_test)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Labels: {y_train.shape}  {y_val.shape}  {y_test.shape}")

if __name__ == "__main__":
    main()
