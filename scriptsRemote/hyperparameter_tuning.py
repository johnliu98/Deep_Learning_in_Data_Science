import pickle

hyper_file = open("hyperparameter_tuning.pkl", "rb")

hyperparameters = pickle.load(hyper_file)
# hyperparameters = hyperparameters.join()
hyperparameters = sorted(hyperparameters, key=lambda x: x[0])

for (acc, params) in hyperparameters:
    print("Accuracy: %.1f ||  Hyperparameters: learning_rate=%.5f, batch_size=%.0f, epochs=%.0f, gruops=%.0f" % \
    (acc, params.learning_rate, params.batch_size, params.epochs, params.groups))
