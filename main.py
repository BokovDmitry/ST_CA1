from src.loaders import cifar10_loader as cf10
from src.loaders import cifar100_loader as cf100
from src import formatter
from src import preprocessor

def main():
    cf10_train = cf10.generate_set()
    cf100_train = cf100.generate_set()

    merged_training_set = formatter.format_sets(cf10_train, cf100_train)

    final_training_set = preprocessor.apply_preprocessing(merged_training_set)

    # cf10_test = cf10.generate_test_set()
    # cf100_test = cf100.generate_test_set()

    # merged_test_set = formatter.merge_sets(cf10_test, cf100_test)



    print(f"Train: {len(final_training_set['labels'])}")
    # print(f"Test: {len(merged_test_set['labels'])}")

if __name__ == "__main__":
    main()
