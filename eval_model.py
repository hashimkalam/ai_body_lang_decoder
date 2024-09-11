from sklearn.metrics import accuracy_score
import pickle

def eval_model(fit_models, x_test, y_test):
    print('Entering eval_model')

    # Evaluate and print accuracy for each model
    for algo, model in fit_models.items():
        yhat = model.predict(x_test)
        accuracy = accuracy_score(y_test, yhat)
        print(f'{algo} Accuracy: {accuracy:.4f}')  # Print accuracy with 4 decimal places

    # Save all models
    with open('body_language_models.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)

    print('All models saved to body_language_models.pkl')

