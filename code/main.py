import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys
import shap
import random
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Concatenate
from tensorflow.keras.layers import LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Embedding, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.tree import export_text

from trace_generator import *
from data_processing import *
from decision_tree import *
from plotting import *


class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

def generate_data(num_cases, model_name, n_gram):
    process_model = build_process_model(model_name)
    folder_name = model_name
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    X, y, class_names, feature_names, feature_indices = generate_processed_data(process_model, categorical_attributes=categorical_attributes, numerical_attributes=numerical_attributes, num_cases=num_cases, n_gram=n_gram, folder_name=folder_name)
    return X, y, class_names, feature_names, feature_indices

# define neural network architecture
def build_nn(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    
    model.compile(optimizer=Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# train and save neural network
def train_nn(X_train, y_train, folder_name=None, model_name="nn.keras"):
    input_dim = X_train.shape[1]  # Number of input features (attributes + events)
    output_dim = y_train.shape[1]  # Number of possible events (classes)
    print("training neural network:")
    print(f"input dimension: {input_dim}")
    print(f"output dimension: {output_dim}")
    model = build_nn(input_dim, output_dim)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    print("--------------------------------------------------------------------------------------------------")
    if folder_name:
        save_nn(model, folder_name, model_name)
    return model

def train_sklearn_dt(X_train, y_train):
    print("training decision tree:")
    # TODO: try various ccp
    #dt = SklearnDecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
    #dt = SklearnDecisionTreeClassifier(max_depth=10, max_leaf_nodes=50, min_samples_leaf=5)
    #dt = SklearnDecisionTreeClassifier(max_depth=10, max_leaf_nodes=50, min_samples_leaf=5, ccp_alpha=0.001)
    #dt = SklearnDecisionTreeClassifier(max_depth=10, min_samples_leaf=5, ccp_alpha=0.001)
    #dt = SklearnDecisionTreeClassifier(min_samples_leaf=5, ccp_alpha=0.001)
    dt = SklearnDecisionTreeClassifier(ccp_alpha=0.01)
    dt.fit(X_train, y_train)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def train_dt(X_train, y_train, folder_name=None, model_name=None, feature_names=None, feature_indices=None, class_names=None):
    dt = train_sklearn_dt(X_train, y_train)
    dt = sklearn_to_custom_tree(dt, feature_names=feature_names, class_names=class_names, feature_indices=feature_indices)
    if model_name:
        save_dt(dt, folder_name, model_name)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def train_custom_dt(X_train, y_train, folder_name=None, model_name=None, feature_names=None, feature_indices=None, class_names=None):
    print("training decision tree:")
    dt = DecisionTreeClassifier(class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    dt.fit(X_train, y_train)
    if model_name:
        save_dt(dt, folder_name, model_name)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def save_nn(model, folder_name, file_name):
    #print(f"saving {file_name}...")
    full_path = os.path.join('models', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    model.save(file_path)

def load_nn(folder_name, file_name):
    #print(f"loading {file_name}...")
    file_name = os.path.join('models', folder_name, file_name)
    model = load_model(file_name)
    return model
    
def save_dt(dt, folder_name, file_name):
    #print(f"saving {file_name}...")
    full_path = os.path.join('models', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    save_tree_to_json(dt, file_path)

def load_dt(folder_name, file_name):
    #print(f"loading {file_name}...")
    file_path = os.path.join('models', folder_name, file_name)
    return load_tree_from_json(file_path)

def evaluate_nn(model, X_test, y_test):
    print("testing nn:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'accuracy: {test_accuracy:.3f}, loss: {test_loss:.3f}')
    print("--------------------------------------------------------------------------------------------------")
    return test_accuracy


def calculate_fairness(nn, X, critical_decisions, feature_indices, class_names, feature_names):
    y = nn.predict(X)
    for decision in critical_decisions:
        previous_feature = "-1. Event = " + decision.previous
        previous_index = feature_names.index(previous_feature)
        filter_mask = X[:, previous_index] == 1
        X_filtered = X[filter_mask]
        y_filtered = y[filter_mask]
        for attribute in decision.attributes:
            for feature_index in feature_indices[attribute]:
                possible_events = decision.possible_events
                metrics = get_metrics(X_filtered, y_filtered, feature_index, possible_events, class_names, feature_names)
                print(f"Metrics for {feature_names[feature_index]}, {possible_events}:")
                for metric in metrics:
                    print(metric)
                print("")

def calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds):
    networks = {
        "Base": nn_base,
        "Enriched": nn_enriched,
        "Modified": nn_modified,
    }
    disp_imp_results = {}
    stat_par_results = {}
    y = {}
    X_adjusted = remove_attribute_features(X, feature_indices, base_attributes)

    for name, nn in networks.items():
        if name == "Base":
            y[name] = nn.predict(X_adjusted)
        else:
            y[name] = nn.predict(X)

    for decision in critical_decisions:
        previous_feature = "-1. Event = " + decision.previous
        previous_index = feature_names.index(previous_feature)
        for attribute in decision.attributes:
            feature_indices_to_use = feature_indices.get(attribute, [])
            for feature_index in feature_indices_to_use:
                for event in decision.possible_events:
                    outer_key = (feature_names[feature_index], event)
                    disp_imp_results[outer_key] = {}
                    stat_par_results[outer_key] = {}
                    for name, nn in networks.items():
                        if event not in class_names:
                            raise ValueError(f"Event '{event}' not found in class_names.")
                        if isinstance(class_names, list):
                            event_index = class_names.index(event)
                        elif isinstance(class_names, np.ndarray):
                            event_index = np.where(class_names == event)[0][0]
                        filter_mask = X[:, previous_index] == 1
                        X_filtered = X[filter_mask]
                        y_filtered = y[name][filter_mask]
                        protected_attribute = X_filtered[:, feature_index]
                        stat_par, disp_imp = get_fairness_metrics(y_filtered, protected_attribute, event_index, feature_names[feature_index], numerical_thresholds)

                        disp_imp_results[outer_key][name] = disp_imp
                        stat_par_results[outer_key][name] = stat_par
                        #print(f"Disparate Impact for {feature_names[feature_index]}, {event}, {name}: {disp_imp}")
                        print(f"Statistical Parity for {feature_names[feature_index]}, {event}, {name}: {stat_par}")

    return stat_par_results, disp_imp_results

def remove_attribute_features(X, feature_indices, base_attributes):
    remove_indices = [idx for attr, indices in feature_indices.items() if attr not in base_attributes for idx in indices]
    return np.delete(X, remove_indices, axis=1)

def get_fairness_metrics(y, protected_attribute, event_index, feature_name, numerical_thresholds):
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    y_binary = np.zeros_like(y)
    y_binary[y == event_index] = 1
    total_amount = np.sum(y_binary == 1)
    if total_amount < 1:
        return 0, 1

    if feature_name in numerical_thresholds:
        # Use threshold to separate groups for numerical attributes
        threshold = numerical_thresholds[feature_name]
        unprivileged_mask = protected_attribute <= threshold
        privileged_mask = protected_attribute > threshold
    else:
        # Default handling for binary attributes
        unprivileged_mask = protected_attribute == 0
        privileged_mask = protected_attribute == 1

    if np.any(unprivileged_mask):
        selection_rate_unprivileged = np.mean(y_binary[unprivileged_mask])
    else:
        selection_rate_unprivileged = 0
    if np.any(privileged_mask):
        selection_rate_privileged = np.mean(y_binary[privileged_mask])
    else:
        selection_rate_privileged = 0

    if selection_rate_privileged == 0:
        disp_impact = float("inf") if selection_rate_unprivileged != 0 else 0
    else:
        disp_impact = selection_rate_unprivileged / selection_rate_privileged

    stat_parity = abs(selection_rate_unprivileged - selection_rate_privileged)
    return stat_parity, disp_impact


def evaluate_dt(dt, X_test, y_test):
    print("testing dt:")
    y_argmax = np.argmax(y_test, axis=1)
    accuracy = dt.score(X_test, y_argmax)
    print(f"Accuracy: {accuracy:.3f}")
    print("")
    dt.visualize()
    print("")
    dt.print_tree_metrics(X_test, y_test, dt.root)
    print("--------------------------------------------------------------------------------------------------")

def distill_nn(nn, X):
    print("distilling nn:")
    softmax_predictions = nn.predict(X)
    y = np.argmax(softmax_predictions, axis=1)
    print("--------------------------------------------------------------------------------------------------")
    return y

def finetune_all(nn, X_train, y_modified, y_distilled_tree, y_distilled, X_test, y_test):
    best_accuracy = 0
    #modes = ['simple', 'changed', 'changed_complete', 'weighted']
    modes = ['simple', 'changed_complete']

    for mode in modes:
        nn_finetuned = clone_model(nn)
        nn_finetuned.set_weights(nn.get_weights())
        nn_finetuned = finetune_nn(nn_finetuned, X_train, y_modified, y_distilled=y_distilled, y_distilled_tree=y_distilled_tree, X_test=X_test, y_test=y_test, mode=mode)
        accuracy_score = evaluate_nn(nn_finetuned, X_test, y_test)
        print(f"Mode: {mode}, with accuracy: {accuracy_score}") 
        if accuracy_score > best_accuracy:
            best_mode = mode
            best_nn = nn_finetuned
            best_accuracy = accuracy_score
        
    print(f"Best mode: {best_mode}, with accuracy: {best_accuracy}") 
    return best_nn


def finetune_nn(nn, X_train, y_modified, y_distilled_tree=None, y_distilled=None, X_test=None, y_test=None, epochs=5, batch_size=32, learning_rate=1e-3, weight=5, mode="simple"):
    # if mode is simple, just train with y_modified
    print(f"Finetuning with mode: {mode}")
    if mode == "simple":
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        if X_test is None:
            nn.fit(X_train, y_modified, epochs=epochs, batch_size=batch_size)
        else:
            nn.fit(X_train, y_modified, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # if mode is changed complete, use the samples that changed value
    elif mode == "changed_complete":
        changed_rows = np.any(y_distilled_tree != y_modified, axis=1)
        changed_indices = np.where(changed_rows)[0]
        y_changed_complete = y_distilled.copy()
        y_changed_complete[changed_indices] = y_modified[changed_indices]
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Changed {len(changed_indices)} out of {len(X_train)} samples")
        if X_test is None:
            nn.fit(X_train, y_changed_complete, epochs=epochs, batch_size=batch_size)
        else:
            nn.fit(X_train, y_changed_complete, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # if mode is changed, just use the samples that changed value
    elif mode == "changed":
        changed_rows = np.any(y_distilled_tree != y_modified, axis=1)
        changed_indices = np.where(changed_rows)[0]
        X_changed = X_train[changed_indices]
        y_changed = y_modified[changed_indices]
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        if X_test is None:
            nn.fit(X_changed, y_changed, epochs=epochs, batch_size=batch_size)
        else:
            nn.fit(X_changed, y_changed, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    # if mode is weighted, use all samples but weigh the changed indices higher
    elif mode == "weighted":
        changed_indices = np.where(y_distilled_tree != y_modified)[0]
        sample_weight = np.ones(len(y_distilled_tree))
        sample_weight[changed_indices] = weight
        print(len(sample_weight))
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        if X_test is None:
            nn.fit(X_train, y_modified, sample_weight=sample_weight, epochs=epochs, batch_size=batch_size)
        else:
            nn.fit(X_train, y_modified, sample_weight=sample_weight, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    else:
        raise(f"mode {mode} doesn't exist!")

    return nn


def find_missing_ids(dt_distilled, dt_modified):
    modified_node_ids = dt_distilled.collect_node_ids()
    distilled_node_ids = dt_modified.collect_node_ids()
    missing_ids = [item for item in distilled_node_ids if item not in modified_node_ids]
    return missing_ids


def print_samples(n, X, y, class_names, feature_names, numerical_attributes):
    if len(feature_names) != X.shape[1]:
        raise ValueError("The length of feature_names must match the number of columns in X.")
    
    for i, sample in enumerate(X[:n]):
        active_indices = np.where(sample != 0)[0]
        active_features = []
        for idx in active_indices:
            feature_name = feature_names[idx]
            if feature_name in numerical_attributes:
                active_features.append(f"{feature_name}={sample[idx]:.2f}")
            else:
                active_features.append(feature_name)
        target_activity = class_names[np.argmax(y[i])]
        print(f"Sample {i}: Features: {', '.join(active_features)}, Target: {target_activity}")


# generates and or processes the data
def run_preprocessing(folder_name, file_name=None, model_name=None, n_gram=3, num_cases=10000, enrichment=False):
    if model_name:
        print(model_name)
        X, y, class_names, feature_names, feature_indices = generate_data(num_cases, model_name, n_gram)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    elif file_name:
        df = load_xes_to_df(file_name, folder_name=folder_name, num_cases=num_cases)
        if enrichment:
            run_enrichment(folder_name)
            df = load_data(folder_name, "df.pkl")
        categorical_attributes, numerical_attributes = get_attributes(folder_name)
        X, y, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    else:
        if enrichment:
            run_enrichment(folder_name)
        df = load_data(folder_name, "df.pkl")
        categorical_attributes, numerical_attributes = get_attributes(folder_name)
        X, y, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    save_data(X_train, folder_name, 'X_train.pkl')
    save_data(X_test, folder_name, 'X_test.pkl')
    save_data(y_train, folder_name, 'y_train.pkl')
    save_data(y_test, folder_name, 'y_test.pkl')
    y_encoded = np.argmax(y_train, axis=1)
    save_data(y_encoded, folder_name, 'y_encoded.pkl')
    save_data(class_names, folder_name, "class_names.pkl")
    save_data(feature_names, folder_name, "feature_names.pkl")
    save_data(feature_indices, folder_name, "feature_indices.pkl")


def run_train_base(folder_name, console_output=True):
    X_train  = load_data(folder_name, "X_train.pkl")
    y_train  = load_data(folder_name, "y_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")

    nn = train_nn(X_train, y_train, folder_name=folder_name)
    y_distilled = distill_nn(nn, X_train)
    save_data(y_distilled, folder_name, "y_distilled.pkl")
    dt_distilled = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)

    if console_output:
        evaluate_nn(nn, X_test, y_test)
        evaluate_dt(dt_distilled, X_test, y_test)


def run_unfair_data_preset():
    domains = ["hospital", "renting", "lending", "hiring"]
    degrees = ["low", "medium", "high"]

    """
    for domain in domains:
        for degree in degrees:
            folder_name = f"{domain}_{degree}"
            file_name = f"{domain}_log_{degree}.xes"
            print(f"Loading XES data for: {folder_name}")
            load_xes_to_df(file_name, folder_name=folder_name)
            print(f"Processing data for: {folder_name}")
            run_preprocessing(folder_name=folder_name, file_name=file_name)
    """

    for domain in domains:
        for degree in degrees:
            folder_name = f"{domain}_{degree}"
            file_name = f"{domain}_log_{degree}.xes"
            print(f"Training for: {folder_name}")
            run_train_base(folder_name)


# executes the complete pipeline
def run_complete(folder_name, model_name=None, file_name=None, n_gram=3, num_cases=1000, preprocessing=False, train_base=False, enrichment=False, retrain=True):
    if preprocessing or enrichment:
        run_preprocessing(folder_name, model_name=model_name, file_name=file_name, n_gram=n_gram, num_cases=num_cases, enrichment=enrichment)
        run_train_base(folder_name=folder_name, console_output=False)
    elif train_base:
        run_train_base(folder_name=folder_name, console_output=False)
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    nn = load_nn(folder_name, "nn.keras")
    dt_distilled = load_dt(folder_name, "dt.json")
    critical_decisions = get_critical_decisions(folder_name)

    print("Base model:")
    evaluate_nn(nn, X_test, y_test)
    evaluate_dt(dt_distilled, X_test, y_test)
    print(f"Critical Decisions: {critical_decisions}")
    nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
    print(f"Nodes to remove accordingly: {nodes_to_remove}")
    print("--------------------------------------------------------------------------------------------------")
    if nodes_to_remove:
        dt_modified = run_modify(folder_name, node_ids=nodes_to_remove, console_output=False, retrain=retrain)
        run_finetuning(folder_name, learning_rate=0.01, console_output=False)
    print("Pipeline done, running analysis...")
    print("--------------------------------------------------------------------------------------------------")
    run_analysis(folder_name)


# runs analysis on finished models
def run_analysis(folder_name):
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")
    critical_decisions = get_critical_decisions(folder_name)
    folder_path = os.path.join('models', folder_name)

    # analyze all neural networks trees
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.keras'):
            print(f"Analyzing model: {file_name}")
            nn = load_nn(folder_name, file_name)
            evaluate_nn(nn, X_test, y_test)
            print("")
            calculate_fairness(nn, X_test, critical_decisions, feature_indices, class_names, feature_names)

    # analyze all decision trees
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            print(f"Analyzing model: {file_name}")
            dt = load_dt(folder_name, file_name)
            evaluate_dt(dt, X_test, y_test)


# runs finetuning for modified base data
def run_finetuning(folder_name, epochs=5, batch_size=32, learning_rate=1e-3, weight=5, console_output=True):
    X_train  = load_data(folder_name, "X_train.pkl")
    y_train  = load_data(folder_name, "y_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    y_modified = load_data(folder_name, "y_modified.pkl")
    y_distilled = load_data(folder_name, "y_distilled.pkl")
    dt_distilled = load_dt(folder_name, "dt.json")
    dt_modified = load_dt(folder_name, "dt_modified.json")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")

    
    nn = load_nn(folder_name, "nn.keras")
    y_train = nn.predict(X_train)
    y_distilled_tree = dt_distilled.predict(X_train)
    y_distilled_tree = to_categorical(y_distilled_tree, num_classes=len(dt_distilled.class_names))
    y_modified = dt_modified.predict(X_train)
    y_modified = to_categorical(y_modified, num_classes=len(dt_modified.class_names))
    nn_changed_complete = finetune_nn(nn, X_train, y_modified, y_distilled=y_train, y_distilled_tree=y_distilled_tree, mode="changed_complete", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    evaluate_nn(nn_changed_complete, X_test, y_test)
    save_nn(nn_changed_complete, folder_name, "nn_changed_complete.keras")

    return

    print("finetuning mode: simple")
    nn = load_nn(folder_name, "nn.keras")
    y_train = dt_distilled.predict(X_train)
    y_train = to_categorical(y_train, num_classes=len(dt_distilled.class_names))
    nn_simple = finetune_nn(nn, X_train, y_modified, y_distilled=y_train, mode="simple", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_simple.keras")
    y_distilled = distill_nn(nn_simple, X_train)
    dt_simple = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_weighted.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("finetuning mode: changed")
    nn = load_nn(folder_name, "nn.keras")
    nn_changed = finetune_nn(nn, X_train, y_modified, y_distilled=y_train, mode="changed", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_changed, folder_name, "nn_changed.keras")
    y_distilled = distill_nn(nn_changed, X_train)
    dt_changed = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_changed.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("finetuning mode: weighted")
    nn = load_nn(folder_name, "nn.keras")
    nn_weighted = finetune_nn(nn, X_train, y_modified, y_distilled=y_train, mode="weighted", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_weighted, folder_name, "nn_weighted.keras")
    y_distilled = distill_nn(nn_weighted, X_train)
    dt_weighted = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_weighted.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)

    if console_output:
        nn = load_nn(folder_name, "nn.keras")
        print("evaluating base nn:")
        evaluate_nn(nn, X_test, y_test)
        print("evaluating distilled dt:")
        evaluate_dt(dt_distilled, X_test, y_test)
        print("evaluating modified dt:")
        evaluate_dt(dt_modified, X_test, y_test)
        print("evaluating mode: simple")
        evaluate_nn(nn_simple, X_test, y_test)
        evaluate_dt(dt_simple, X_test, y_test)
        print("evaluating mode: changed")
        evaluate_nn(nn_changed, X_test, y_test)
        evaluate_dt(dt_changed, X_test, y_test)
        print("evaluating mode: weighted")
        evaluate_nn(nn_weighted, X_test, y_test)
        evaluate_dt(dt_weighted, X_test, y_test)

        y_modified = dt_modified.predict(X_test)
        y_modified = to_categorical(y_modified, num_classes=len(dt_modified.class_names))
        print("evaluating unfair:")
        changed_indices = np.where(y_test != y_modified)[0]
        X = X_test[changed_indices]
        y = y_test[changed_indices]
        evaluate_nn(nn_weighted, X, y)
        print("evaluating fair:")
        changed_indices = np.where(y_test == y_modified)[0]
        X = X_test[changed_indices]
        y = y_test[changed_indices]
        evaluate_nn(nn_weighted, X, y)


def run_modify(folder_name, node_ids=[], console_output=True, retrain=False):
    X_train  = load_data(folder_name, "X_train.pkl")
    y_train  = load_data(folder_name, "y_encoded.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    dt_distilled = load_dt(folder_name, "dt.json")

    for node_id in node_ids:
        if retrain:
            dt_distilled.delete_node(X_train, y_train, node_id)
        else:
            dt_distilled.delete_branch(node_id)
    save_dt(dt_distilled, folder_name, "dt_modified.json")
    if console_output:
        evaluate_dt(dt_distilled, X_test, y_test)

    y_modified = dt_distilled.predict(X_train)
    y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))
    save_data(y_encoded, folder_name, "y_modified.pkl")

    return dt_distilled

def run_demo(folder_name, n_gram=3, num_cases=1000, preprocessing=False):
    run_ablation_decisions()
    sys.exit()
    run_4_cases()
    male_shapley_scores = load_data(folder_name, "male_shapley_scores.pkl")
    female_shapley_scores = load_data(folder_name, "female_shapley_scores.pkl")
    male_shapley_scores_modified = load_data(folder_name, "male_shapley_scores_modified.pkl")
    female_shapley_scores_modified = load_data(folder_name, "female_shapley_scores_modified.pkl")
    plot_shapley(male_shapley_scores, male_shapley_scores_modified, folder_name, "shapley_male.png")
    plot_shapley(female_shapley_scores, female_shapley_scores_modified, folder_name, "shapley_female.png")

def load_bpi_A():
    folder_name = "bpi_A"
    df = load_xes_to_df("bpi_2012.xes")
    df = df[df['activity'].str.startswith('A_')]
    rules = get_rules(folder_name)
    df = enrich_df(df, rules, folder_name)
    print(df.head(20))
    save_data(df, folder_name, "df.pkl")


def evaluate_all():
    folder_names = ["cs", "hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender", "bpi"]
    retrain_options = [True, False]

    for folder_name in folder_names:
        for retrain_option in retrain_options:
            run_k_fold(folder_name, folds=5, n_gram=3, retrain=retrain_option, finetune_mode=None)

    run_ablation_bias()
    run_ablation_attributes()
    run_ablation_decisions()
    run_ablation_decisions_biased()


def run_4_cases():
    folder_names = ["hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender"]

    for folder_name in folder_names:
        load_xes_to_df("hospital_billing", folder_name=folder_name)
        run_enrichment(folder_name)
        #run_k_fold(folder_name)


def run_enrichment(folder_name):
    print("Running enrichment...")
    df = load_data(folder_name, "df.pkl")
    rules = get_rules(folder_name)
    df = enrich_df(df, rules, folder_name)
    print(df.head(20))
    save_data(df, folder_name, "df.pkl")


def run_k_fold(folder_name, folds=5, n_gram=3, retrain=True, finetune_mode=None):
    df = load_data(folder_name, "df.pkl")
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    if "bpi" in folder_name:
        categorical_attributes_base, numerical_attributes_base = [], ["case:AMOUNT_REQ", "time_delta"]
    else:
        categorical_attributes_base, numerical_attributes_base = get_attributes("")
    base_attributes = categorical_attributes_base + numerical_attributes_base
    critical_decisions = get_critical_decisions(folder_name)
    k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, folds=folds, n_gram=n_gram, retrain=retrain, finetune_mode=finetune_mode, folder_name=folder_name)


def k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, folds=5, n_gram=3, retrain=True, finetune_mode=None, folder_name=None):
    base_accuracy_values = []
    enriched_accuracy_values = []
    modified_accuracy_values = []
    node_counts = []
    stat_par_results = {}
    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, categorical_attributes)
    feature_names = create_feature_names(class_names, attribute_pools, numerical_attributes, n_gram)
    feature_indices = create_feature_indices(class_names, attribute_pools, numerical_attributes, n_gram)
    print(class_names)
    print(feature_names)
    print(feature_indices)

    folds = k_fold_cross_validation(df, categorical_attributes, numerical_attributes, critical_decisions, n_gram=3, k=folds)

    for i, (X_train, y_train, X_test, y_test, numerical_thresholds) in tqdm(enumerate(folds), desc="evaluating model:"):
        # evaluating the base model
        X_train_base = remove_attribute_features(X_train, feature_indices, base_attributes)
        X_test_base = remove_attribute_features(X_test, feature_indices, base_attributes)
        nn_base = train_nn(X_train_base, y_train)
        base_accuracy = evaluate_nn(nn_base, X_test_base, y_test)
        base_accuracy_values.append(base_accuracy)

        # evaluating the enriched model
        nn_enriched = train_nn(X_train, y_train)
        y_distilled = nn_enriched.predict(X_train)
        y_encoded = np.argmax(y_distilled, axis=1)

        dt_distilled = train_dt(X_train, y_encoded, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
        enriched_accuracy = evaluate_nn(nn_enriched, X_test, y_test)
        enriched_accuracy_values.append(enriched_accuracy)

        evaluate_dt(dt_distilled, X_test, y_test)
        num_nodes = dt_distilled.count_nodes()
        node_counts.append(num_nodes)

        # modifying the distilled model
        nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
        y_distilled_tree = dt_distilled.predict(X_train)
        y_distilled_tree = to_categorical(y_distilled_tree, num_classes=len(dt_distilled.class_names))

        # prepare modified model
        nn_modified = clone_model(nn_enriched)
        nn_modified.set_weights(nn_enriched.get_weights())
        if nodes_to_remove:
            print(f"Removing nodes: {nodes_to_remove}")
            if retrain:
                y_encoded = np.argmax(y_train, axis=1)
                for node_id in nodes_to_remove:
                    dt_distilled.delete_node(X_train, y_encoded, node_id)
            else:
                for node_id in nodes_to_remove:
                    dt_distilled.delete_branch(node_id)
            evaluate_dt(dt_distilled, X_test, y_test)
            y_modified = dt_distilled.predict(X_train)
            y_modified = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))

            # finetuning and evaluating the changed model
            if finetune_mode:
                nn_modified = finetune_nn(nn_modified, X_train, y_modified, y_distilled=y_distilled, y_distilled_tree=y_distilled_tree, X_test=X_test, y_test=y_test, mode=finetune_mode)
            else:
                nn_modified = finetune_all(nn_modified, X_train, y_modified, y_distilled_tree, y_distilled, X_test, y_test)
            modified_accuracy = evaluate_nn(nn_modified, X_test, y_test)
        else:
            print(f"No nodes to remove!")
            modified_accuracy = enriched_accuracy
        modified_accuracy_values.append(modified_accuracy)

        print(f"base accuracy: {base_accuracy}, enriched accuracy: {enriched_accuracy}, modified accuracy: {modified_accuracy}")
        calculate_fairness(nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names)
        stat_par_result, _ = calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes, numerical_thresholds)
        for outer_key, outer_value in stat_par_result.items():
            if i == 0:
                stat_par_results[outer_key] = {}
            for inner_key, inner_value in outer_value.items():
                if i == 0:
                    stat_par_results[outer_key][inner_key] = []
                stat_par_results[outer_key][inner_key].append(inner_value)
        print("--------------------------------------------------------------------------------------------------")

    node_counts_array = np.array(node_counts)
    average_nodes = np.mean(node_counts_array)
    std_dev_nodes = np.std(node_counts_array)
    print(f"Average number of nodes: {average_nodes:.2f}")
    print(f"Standard deviation of nodes: {std_dev_nodes:.2f}")
    if folder_name:
        save_data(node_counts_array, folder_name, "node_counts.pkl")
        save_data(base_accuracy_values, folder_name, "base_accuracy_values.pkl")
        save_data(enriched_accuracy_values, folder_name, "enriched_accuracy_values.pkl")
        save_data(modified_accuracy_values, folder_name, "modified_accuracy_values.pkl")
        save_data(stat_par_results, folder_name, "stat_par_results.pkl")
        title = f"Accuracy Distribution of {folder_name}"
        plot_distribution(base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, folder_name, title, "Accuracy")
        plot_all_parity(stat_par_results, folder_name)
    return base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, stat_par_results



def run_ablation_decisions():
    numerical_attributes = ["time_delta"]
    base_attributes = ["time_delta"]
    bias = 0.7

    x_values = []
    base_accuracies = []
    enriched_accuracies = []
    modified_accuracies = []
    base_fairness = []
    enriched_fairness = []
    modified_fairness = []

    for num_decisions in np.arange(4, 7):
        if num_decisions == 0:
            num_decisions = 2
        print(f"Analyzing num_decisions: {num_decisions}")
        critical_decisions = []
        categorical_attributes = ["a_0"]
        process_model = build_process_model("ablation_num_decisions")
        process_model.add_categorical_attribute("a_0", [("A", 0.5), ("B", 0.5)])
        process_model.add_activity("A_0")
        process_model.add_activity("B_0")
        process_model.add_activity("C_0")
        process_model.add_activity("D_0")
        conditions_top = {("a_0", "A"): bias, ("a_0", "B"): 1-bias}
        conditions_bottom = {("a_0", "A"): 1-bias, ("a_0", "B"): bias}
        process_model.add_transition("start", "A_0", conditions=conditions_top)
        process_model.add_transition("start", "B_0", conditions=conditions_bottom)
        process_model.add_transition("A_0", "C_0", conditions=conditions_top)
        process_model.add_transition("A_0", "D_0", conditions=conditions_bottom)
        process_model.add_transition("B_0", "C_0", conditions=conditions_top)
        process_model.add_transition("B_0", "D_0", conditions=conditions_bottom)
        critical_decisions.append(Decision(attributes=["a_0"], possible_events=["A_0", "B_0"], to_remove=True, previous="start"))
        activity_name_c = ""
        activity_name_d = ""

        for n in range(1, num_decisions):
            activity_name_a = f"A_{n}"
            activity_name_b = f"B_{n}"
            activity_name_c = f"C_{n}"
            activity_name_d = f"D_{n}"
            previous_activity_name_c = f"C_{n-1}"
            previous_activity_name_d = f"D_{n-1}"
            attribute_name = f"a_{n}"
            conditions_top = {(attribute_name, "A"): bias, (attribute_name, "B"): 1-bias}
            conditions_bottom = {(attribute_name, "A"): 1-bias, (attribute_name, "B"): bias}

            process_model.add_activity(activity_name_a)
            process_model.add_activity(activity_name_b)
            process_model.add_activity(activity_name_c)
            process_model.add_activity(activity_name_d)
            process_model.add_categorical_attribute(attribute_name, [("A", 0.5), ("B", 0.5)])
            categorical_attributes.append(attribute_name)

            process_model.add_transition(previous_activity_name_c, activity_name_a, conditions=conditions_top)
            process_model.add_transition(previous_activity_name_c, activity_name_b, conditions=conditions_bottom)
            process_model.add_transition(previous_activity_name_d, activity_name_a, conditions=conditions_top)
            process_model.add_transition(previous_activity_name_d, activity_name_b, conditions=conditions_bottom)
            process_model.add_transition(activity_name_a, activity_name_c, conditions=conditions_top)
            process_model.add_transition(activity_name_a, activity_name_d, conditions=conditions_bottom)
            process_model.add_transition(activity_name_b, activity_name_c, conditions=conditions_top)
            process_model.add_transition(activity_name_b, activity_name_d, conditions=conditions_bottom)

            critical_decisions.append(Decision(attributes=[attribute_name], possible_events=[activity_name_a, activity_name_b], to_remove=True, previous=previous_activity_name_c))
            critical_decisions.append(Decision(attributes=[attribute_name], possible_events=[activity_name_a, activity_name_b], to_remove=True, previous=previous_activity_name_d))

        process_model.add_transition(activity_name_c, "end", conditions={})
        process_model.add_transition(activity_name_d, "end", conditions={})

        trace_generator = TraceGenerator(process_model=process_model)
        cases = trace_generator.generate_traces(num_cases=1000)
        df = cases_to_dataframe(cases)
        
        base_accuracy, enriched_accuracy, modified_accuracy, stat_par_results = k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, folds=5, n_gram=3, retrain=False, finetune_mode=None, folder_name=None)

        x_values.append(num_decisions)
        base_accuracies.append(np.mean(base_accuracy))
        enriched_accuracies.append(np.mean(enriched_accuracy))
        modified_accuracies.append(np.mean(modified_accuracy))
        fairness = {"Base": [], "Enriched": [], "Modified": []}
        for n in range(num_decisions):
            fairness_n = stat_par_results[(f"a_{n} = A", f"A_{n}")]
            fairness["Base"].append(np.mean(fairness_n["Base"]))
            fairness["Enriched"].append(np.mean(fairness_n["Enriched"]))
            fairness["Modified"].append(np.mean(fairness_n["Modified"]))
        base_fairness.append(np.mean(np.array(fairness["Base"])))
        enriched_fairness.append(np.mean(np.array(fairness["Enriched"])))
        modified_fairness.append(np.mean(np.array(fairness["Modified"])))

    x_values = np.array(x_values)
    x_values *= 2
    plot_ablation(x_values, base_accuracies, enriched_accuracies, modified_accuracies, "Accuracy for Number of Decisions", "Number of Decisions", "Accuracy", "ablation_decisions_biased")
    plot_ablation(x_values, base_fairness, enriched_fairness, modified_fairness, "Demographic Parity for Number of Decisions", "Number of Decisions", "Demographic Parity", "ablation_decisions_biased")

def run_ablation_attributes():
    numerical_attributes = ["time_delta", "age"]
    base_attributes = ["time_delta", "age"]

    x_values = []
    base_accuracies = []
    enriched_accuracies = []
    modified_accuracies = []
    base_fairness = []
    enriched_fairness = []
    modified_fairness = []

    for num_attributes in np.arange(0, 21, 5):
        if num_attributes == 0:
            num_attributes = 1
        print(f"Analyzing num_attributes: {num_attributes}")
        process_model = build_process_model("ablation_num_attributes")
        categorical_attributes = []
        critical_decisions = []

        for n in range(num_attributes):
            attribute_name = f"a_{n}"
            bias = 0.7
            process_model.add_categorical_attribute(attribute_name, [("A", 0.5), ("B", 0.5)])
            categorical_attributes.append(attribute_name)
            critical_decisions.append(Decision(attributes=[attribute_name], possible_events=["collect history", "refuse screening"], to_remove=True, previous="asses eligibility"))

            process_model.add_transition("asses eligibility", "collect history", conditions={
                (attribute_name, "A"): bias,
                (attribute_name, "B"): 1-bias,  
            })
            process_model.add_transition("asses eligibility", "refuse screening", conditions={
                (attribute_name, "A"): 1-bias,
                (attribute_name, "B"): bias,  
            })

            flip = random.randint(0,1)
            if flip:
                bias = 1-bias

            process_model.add_transition("collect history", "prostate screening", conditions={
                (attribute_name, "A"): bias,
                (attribute_name, "B"): 1-bias,  
            })
            process_model.add_transition("collect history", "mammary screening", conditions={
                (attribute_name, "A"): 1-bias,
                (attribute_name, "B"): bias,  
            })

        trace_generator = TraceGenerator(process_model=process_model)
        cases = trace_generator.generate_traces(num_cases=10000)
        df = cases_to_dataframe(cases)
        
        base_accuracy, enriched_accuracy, modified_accuracy, stat_par_results = k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, folds=5, n_gram=3, retrain=False, finetune_mode=None, folder_name=None)

        x_values.append(num_attributes)
        base_accuracies.append(np.mean(base_accuracy))
        enriched_accuracies.append(np.mean(enriched_accuracy))
        modified_accuracies.append(np.mean(modified_accuracy))
        fairness = {"Base": [], "Enriched": [], "Modified": []}
        for n in range(num_attributes):
            fairness_n = stat_par_results[(f"a_{n} = A", "collect history")]
            fairness["Base"].append(np.mean(fairness_n["Base"]))
            fairness["Enriched"].append(np.mean(fairness_n["Enriched"]))
            fairness["Modified"].append(np.mean(fairness_n["Modified"]))
        base_fairness.append(np.mean(np.array(fairness["Base"])))
        enriched_fairness.append(np.mean(np.array(fairness["Enriched"])))
        modified_fairness.append(np.mean(np.array(fairness["Modified"])))

    plot_ablation(x_values, base_accuracies, enriched_accuracies, modified_accuracies, "Accuracy for Number of Attributes", "Number of Attributes", "Accuracy", "ablation_attributes")
    plot_ablation(x_values, base_fairness, enriched_fairness, modified_fairness, "Demographic Parity for Number of Attributes", "Number of Attributes", "Demographic Parity", "ablation_attributes")


def run_ablation_bias():
    critical_decisions = []
    critical_decisions.append(Decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True, previous="asses eligibility"))
    critical_decisions.append(Decision(attributes=["gender"], possible_events=["prostate screening", "mammary screening"], to_remove=False, previous="collect history"))
    categorical_attributes = ["gender"]
    numerical_attributes = ["time_delta", "age"]
    base_attributes = ["time_delta", "age"]

    x_values = []
    base_accuracies = []
    enriched_accuracies = []
    modified_accuracies = []
    base_fairness = []
    enriched_fairness = []
    modified_fairness = []

    for bias in np.arange(0.5, 1.05, 0.05):
        print(f"Analyzing Bias strength: {bias}")
        process_model = build_process_model("ablation_strength")
        process_model.add_transition("asses eligibility", "collect history", conditions={
            ("gender", "male"): bias,
            ("gender", "female"): 1-bias,  
        })
        process_model.add_transition("asses eligibility", "refuse screening", conditions={
            ("gender", "male"): 1-bias,
            ("gender", "female"): bias,  
        })
        trace_generator = TraceGenerator(process_model=process_model)
        cases = trace_generator.generate_traces(num_cases=10000)
        df = cases_to_dataframe(cases)
        
        base_accuracy, enriched_accuracy, modified_accuracy, stat_par_results = k_fold_evaluation(df, critical_decisions, categorical_attributes, numerical_attributes, base_attributes, folds=5, n_gram=3, retrain=False, finetune_mode=None, folder_name=None)

        x_values.append(bias)
        base_accuracies.append(np.mean(base_accuracy))
        enriched_accuracies.append(np.mean(enriched_accuracy))
        modified_accuracies.append(np.mean(modified_accuracy))
        print(stat_par_results)
        fairness = stat_par_results[("gender = male", "collect history")]
        base_fairness.append(np.mean(fairness["Base"]))
        enriched_fairness.append(np.mean(fairness["Enriched"]))
        modified_fairness.append(np.mean(fairness["Modified"]))

    plot_ablation(x_values, base_accuracies, enriched_accuracies, modified_accuracies, "Accuracy for Bias Strength", "Bias", "Accuracy", "ablation_bias")
    plot_ablation(x_values, base_fairness, enriched_fairness, modified_fairness, "Demographic Parity for Bias Strength", "Bias", "Demographic Parity", "ablation_bias")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model to use (default: None)')
    parser.add_argument('--file_name', type=str, default=None, help='Name of the file to use (default: None)')
    parser.add_argument('--folder_name', type=str, default=None, help='Name of the folder to use (default: same as model_name)')
    parser.add_argument('--finetune_mode', choices=['simple', 'changed', 'changed_complete', 'weighted'], type=str, default=None, help='Mode used for fine tuning')
    parser.add_argument('--mode', choices=['a', 'c', 'd', 'e', 'f', 'i', 'k', 'm', 'p', 't'], default='c',
                        help="Choose 'complete' to run full pipeline or 'preprocessed' to use preprocessed data (default: complete)")
    parser.add_argument('--n_gram', type=int, default=3, help='Value for n-gram (default: 3)')
    parser.add_argument('--num_cases', type=int, default=1000, help='Number of cases to process (default: 1000)')
    parser.add_argument('--no-save', dest='save', action='store_false', 
                        help='Disable saving the results (default: results will be saved)')
    parser.add_argument('--e', dest='enrichment', action='store_true', 
                        help='Enrich original data')
    parser.add_argument('--t', dest='train_base', action='store_true', 
                        help='Train base models')
    parser.add_argument('--p', dest='preprocessing', action='store_true', 
                        help='Generate new data')
    parser.add_argument('--r', dest='retrain', action='store_true', 
                        help='Retrain deleted nodes')
    parser.add_argument("node_ids", type=int, nargs="*", help="List of node_ids")
    args = parser.parse_args()
    if args.folder_name is None:
        args.folder_name = args.model_name
    
    # Check which mode is selected and run the corresponding function
    if args.mode == 'a':
        run_analysis(args.folder_name)
    elif args.mode == 'c':
        run_complete(args.folder_name, model_name=args.model_name, file_name=args.file_name, n_gram=args.n_gram, num_cases=args.num_cases, preprocessing=args.preprocessing, train_base=args.train_base, enrichment=args.enrichment, retrain=args.retrain)
    elif args.mode == 'd':
        run_demo(args.folder_name, n_gram=args.n_gram, num_cases=args.num_cases, preprocessing=args.preprocessing)
    elif args.mode == 'e':
        run_enrichment(args.folder_name)
    elif args.mode == 'f':
        run_finetuning(args.folder_name)
    elif args.mode == 'i':
        run_interactive()
    elif args.mode == 'k':
        run_k_fold(args.folder_name, retrain=args.retrain, finetune_mode=args.finetune_mode)
    elif args.mode == 'm':
        run_modify(args.folder_name, node_ids=args.node_ids, retrain=args.retrain)
    elif args.mode == 'p':
        run_preprocessing(args.folder_name, model_name=args.model_name, file_name=args.file_name, n_gram=args.n_gram, num_cases=args.num_cases)
    elif args.mode == 't':
        run_train_base(args.folder_name)

if __name__ == "__main__":
    main()
    print("Done and dusted!")