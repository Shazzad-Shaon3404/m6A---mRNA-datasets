import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

# Load  dataset
data_path = '/content/drive/MyDrive/a new RNA m6A/m6A all/dataset/main data for merged/100-XGB-Features.csv'
data = pd.read_csv(data_path)


X = data.drop('Target', axis=1).values
y = data['Target'].values


X = X.reshape(X.shape[0], 100, 1)


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


accuracies = []
specificities = []
sensitivities = []
mccs = []
kappas = []
auc_scores = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build the CNN-Attention model
    cnn_input = Input(shape=(100, 1))
    cnn_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)

    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Dropout(0.2)(cnn_layer)
    cnn_layer = Conv1D(filters=64, kernel_size=5, activation='relu')(cnn_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Dropout(0.5)(cnn_layer)
    cnn_layer = Conv1D(filters=128, kernel_size=7, activation='relu')(cnn_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Dropout(0.5)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)

    # Attention mechanism
    attention_layer = Attention()([cnn_layer, cnn_layer])

    # Concatenate CNN and attention outputs
    combined = tf.keras.layers.concatenate([cnn_layer, attention_layer])

    # Fully connected layers
    combined = Dense(128, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)

    # Compile the model
    modelca = Model(inputs=cnn_input, outputs=output)
    modelca .compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy', AUC()])

    # Train the model
    modelca .fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)



    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred)

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    mcc = matthews_corrcoef(y_test, y_pred_classes)
    kappa = cohen_kappa_score(y_test, y_pred_classes)
    auc_score = roc_auc_score(y_test, y_pred)

    # Store metrics
    accuracies.append(accuracy)
    specificities.append(specificity)
    sensitivities.append(sensitivity)
    mccs.append(mcc)
    kappas.append(kappa)
    auc_scores.append(auc_score)

# Calculate mean and standard deviation for metrics
mean_accuracy = np.mean(accuracies)
mean_specificity = np.mean(specificities)
mean_sensitivity = np.mean(sensitivities)
mean_mcc = np.mean(mccs)
mean_kappa = np.mean(kappas)
mean_auc_score = np.mean(auc_scores)
# Calculate standard deviations (std) for metrics
std_accuracy = np.std(accuracies)
std_specificity = np.std(specificities)
std_sensitivity = np.std(sensitivities)
std_mcc = np.std(mccs)
std_kappa = np.std(kappas)
std_auc_score = np.std(auc_scores)

print("Std Accuracy:", std_accuracy)
print("Std Specificity:", std_specificity)
print("Std Sensitivity:", std_sensitivity)
print("Std MCC Score:", std_mcc)
print("Std Kappa Score:", std_kappa)
print("Std AUC Score:", std_auc_score)
print("Mean Accuracy:", mean_accuracy)
print("Mean Specificity:", mean_specificity)
print("Mean Sensitivity:", mean_sensitivity)
print("Mean MCC Score:", mean_mcc)
print("Mean Kappa Score:", mean_kappa)
print("Mean AUC Score:", mean_auc_score)

