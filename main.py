from src.loaders import cifar10_loader as cf10
from src.loaders import cifar100_loader as cf100
import formatter

def main():
    cf10_train = cf10.generate_set()
    cf100_train = cf100.generate_set()

    merged_set = formatter.format_sets(cf10_train, cf100_train)

    print(len(merged_set.get('labels')))




if __name__ == "__main__":
    main()
