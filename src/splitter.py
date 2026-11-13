from sklearn.model_selection import train_test_split

VALIDATION_SIZE = 0.1

def split_validation_set(data, validation_size=VALIDATION_SIZE):
    X_train, X_val, y_train, y_val = train_test_split(
        data['data'], data['labels'],
        test_size=validation_size,
        stratify=data['labels'],
        random_state=0,
    )
    return X_train, y_train, X_val, y_val
