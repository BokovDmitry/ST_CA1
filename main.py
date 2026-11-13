from src.loaders import cifar10_loader as cf10
from src.loaders import cifar100_loader as cf100
from src.loaders.class_names import CLASS_NAMES
from src import formatter
from src import preprocessor
from src import splitter
from src import explorer

def main():
    num_classes = len(CLASS_NAMES)

    cf10_train = cf10.generate_set()
    cf100_train = cf100.generate_set()

    merged_training_set = formatter.format_sets(cf10_train, cf100_train)

    cf10_test = cf10.generate_test_set()
    cf100_test = cf100.generate_test_set()

    merged_test_set = formatter.format_sets(cf10_test, cf100_test)

    X_train, y_train, X_val, y_val = splitter.split_validation_set(merged_training_set)
    X_test, y_test = merged_test_set['data'], merged_test_set['labels']

    explorer.summarise_set('Train', X_train, y_train)
    explorer.summarise_set('Validation', X_val, y_val)
    explorer.summarise_set('Test', X_test, y_test)
    num_of_each_class = explorer.show_training_samples(num_classes, X_train, y_train)
    explorer.plot_sample_distribution(num_classes, num_of_each_class)

    X_train = preprocessor.apply_preprocessing(X_train)
    X_val = preprocessor.apply_preprocessing(X_val)
    X_test = preprocessor.apply_preprocessing(X_test)

    preprocessor.examine_random_training_image_after_preprocessing(X_train)

    y_train, y_val, y_test = formatter.one_hot_encode(num_classes, y_train, y_val, y_test)

    print(f"After preprocessing - Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

if __name__ == "__main__":
    main()
