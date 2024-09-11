from dataset_creation import dataset_creation
from train_model import train_model
from eval_model import eval_model
from test_model import test_model

def main():
    dataset_creation("--class--")
    fit_models, x_test, y_test = train_model()
    eval_model(fit_models=fit_models, x_test=x_test, y_test=y_test)
    test_model()

if __name__ == "__main__":
    main()