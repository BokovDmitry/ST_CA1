import matplotlib.pyplot as plt


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    history = model.fit(X_train, y_train, batch_size=50, epochs=10, validation_data=(X_val, y_val), verbose=1, shuffle=1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])
