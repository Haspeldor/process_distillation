import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential, load_model
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


def generate_data(num_cases, model_name, n_gram):
    process_model = build_process_model(model_name)
    folder_name = model_name
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices, critical_decisions = generate_processed_data(process_model, categorical_attributes=categorical_attributes, numerical_attributes=numerical_attributes, num_cases=num_cases, n_gram=n_gram, folder_name=folder_name)
    return X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices, critical_decisions

# define neural network architecture
def build_nn(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    
    model.compile(optimizer=Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def build_transformer_model(vocab_size, max_seq_len, padding_value, num_categorical=0, num_numerical=0, embed_dim=32, num_heads=2, ff_dim=64, dropout_rate=0.1):
    """
    Creates a transformer model for next activity prediction.

    Args:
        vocab_size (int): Number of unique activities.
        max_seq_len (int): Maximum sequence length.
        num_categorical (int): Number of categorical features.
        num_numerical (int): Number of numerical features.
        embed_dim (int): Embedding dimension for activities.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward network dimension.
        dropout_rate (float): Dropout rate.

    Returns:
        tf.keras.Model: Compiled transformer model.
    """
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    masked_input = Masking(mask_value=padding_value)(activity_input)
    activity_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(masked_input)

    # Positional Encoding
    position_input = tf.range(start=0, limit=max_seq_len, delta=1)
    position_embedding = Embedding(input_dim=max_seq_len, output_dim=embed_dim)(position_input)
    x = activity_embedding + position_embedding

    # Transformer Encoder Block with Mask
    attention_mask = tf.cast(tf.not_equal(activity_input, padding_value), tf.float32)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
    x, x, attention_mask=attention_mask
    )
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Fully Connected Layers
    ff_output = Dense(ff_dim, activation='relu')(attn_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(embed_dim)(ff_output)
    x = LayerNormalization(epsilon=1e-6)(attn_output + ff_output)

    # Output layer
    output = Dense(vocab_size, activation='softmax', name='output')(x)

    # Compile Model
    inputs = [activity_input]  # Add other categorical/numerical inputs if needed
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_accuracy])

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
    #dt = SklearnDecisionTreeClassifier(max_depth=15, min_samples_leaf=5)
    dt = SklearnDecisionTreeClassifier(max_depth=10, max_leaf_nodes=50, min_samples_leaf=5, ccp_alpha=0.01)
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
    print(f'accuracy: {test_accuracy:.2f}, loss: {test_loss:.2f}')
    print("--------------------------------------------------------------------------------------------------")

def print_metrics_nn(nn, dt, X, node_ids):
    class_names = dt.class_names
    feature_names = dt.feature_names
    _ = np.zeros_like(X)

    for node_id in node_ids:
        X, _ = dt._get_data_for_subtree(X, _, node_id)
        node = dt.find_node(node_id)
        feature_index = node.feature_index
        possible_events = dt.collect_events(node_id)
        y = nn.predict(X)
        y = np.argmax(y, axis=1)
        print(isinstance(X, np.ndarray))
        print(isinstance(y, np.ndarray))

        metrics = get_metrics(X, y, feature_index, possible_events, class_names, feature_names)
        print(f"Metrics for modified node {node_id}:")
        for metric in metrics:
            print(metric)

def calculate_fairness(nn, X, critical_decisions, feature_indices, class_names, feature_names):
    y = nn.predict(X)
    for decision in critical_decisions:
        for attribute in decision.attributes:
            for feature_index in feature_indices[attribute]:
                possible_events = decision.possible_events
                metrics = get_metrics(X, y, feature_index, possible_events, class_names, feature_names)
                print(f"Metrics for {feature_names[feature_index]}, {possible_events}:")
                for metric in metrics:
                    print(metric)
                print("")

def evaluate_dt(dt, X_test, y_test):
    print("testing dt:")
    y_argmax = np.argmax(y_test, axis=1)
    accuracy = dt.score(X_test, y_argmax)
    print(f"Accuracy: {accuracy:.2f}")
    print("")
    dt.visualize()
    print("")
    dt.print_tree_metrics(X_test, y_test, dt.root)
    print("--------------------------------------------------------------------------------------------------")

def evaluate_sklearn_dt(dt, X_test, y_test, y_distilled, class_names=None, feature_names=None):
    print("testing dt:")
    y_argmax = np.argmax(y_test, axis=1)
    accuracy = dt.score(X_test, y_argmax)
    seen = set()
    sklearn_class_names = [
        class_names[i] for i in y_distilled if i < len(class_names) and not (i in seen or seen.add(i))
    ]
    sklearn_class_names.sort()
    print(f"Test set accuracy: {accuracy:.2f}")
    tree_text = export_text(dt, class_names=sklearn_class_names, feature_names=feature_names)
    print(tree_text)

def distill_nn(nn, X):
    print("distilling nn:")
    softmax_predictions = nn.predict(X)
    y = np.argmax(softmax_predictions, axis=1)
    print("--------------------------------------------------------------------------------------------------")
    return y

def masked_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    loss *= mask  # Only consider non-padding tokens
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    correct_predictions = tf.cast(
        tf.equal(tf.argmax(y_pred, axis=-1), tf.cast(y_true, tf.int64)), dtype=tf.float32
    )
    correct_predictions *= mask  # Only consider non-padding tokens
    return tf.reduce_sum(correct_predictions) / tf.reduce_sum(mask)


def finetune_nn(nn, X, y_modified, y_train=[], epochs=5, batch_size=32, learning_rate=1e-5, weight=5, mode="simple"):
    # if mode is simple, just train with y_modified
    if mode == "simple":
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        nn.fit(X, y_modified, epochs=epochs, batch_size=batch_size)

    # if mode is changed, just use the samples that changed value
    elif mode == "changed":
        changed_rows = np.any(y_train != y_modified, axis=1)
        changed_indices = np.where(changed_rows)[0]
        X_changed = X[changed_indices]
        y_changed = y_modified[changed_indices]
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        nn.fit(X_changed, y_changed, epochs=epochs, batch_size=batch_size)
    
    # if mode is weighted, use all samples but weigh the changed indices higher
    elif mode == "weighted":
        changed_indices = np.where(y_train != y_modified)[0]
        sample_weight = np.ones(len(y_train))
        sample_weight[changed_indices] = weight
        print(len(sample_weight))
        nn.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        nn.fit(X, y_modified, sample_weight=sample_weight, epochs=epochs, batch_size=batch_size)
    
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


def get_attributes(folder_name):
    categorical_attributes = ["day_of_week"]
    numerical_attributes = ["time_delta", "time_of_day"]

    if "hiring" in folder_name:
        categorical_attributes += ["case:citizen", "case:german speaking", "case:gender"]
        numerical_attributes += ["case:age", "case:yearsOfEducation"]
    elif "hospital" in folder_name:
        categorical_attributes += ["case:citizen", "case:german speaking", "case:gender", "case:private_insurance", "case:underlying_condition"]
        numerical_attributes += ["case:age"]
    elif "lending" in folder_name:
        categorical_attributes += ["case:citizen", "case:german speaking", "case:gender"]
        numerical_attributes += ["case:age", "case:yearsOfEducation", "case:CreditScore"]
    elif "renting" in folder_name:
        categorical_attributes += ["case:citizen", "case:german speaking", "case:gender", "case:married"]
        numerical_attributes += ["case:age", "case:yearsOfEducation"]
    elif "cc_n"  in folder_name:
        categorical_attributes += ["gender"]
        numerical_attributes += ["age"]

    return categorical_attributes, numerical_attributes


def get_critical_decisions(folder_name):
    critical_decisions = []
    if "hiring" in folder_name:
        critical_decisions.append(Decision(attributes=["case:age", "case:citizen", "case:german speaking", "case:gender"], possible_events=["Application Rejected"], to_remove=True))
        critical_decisions.append(Decision(attributes=["case:yearsOfEducation"], possible_events=["Application Rejected"], to_remove=False))
    elif "hospital" in folder_name:
        critical_decisions.append(Decision(attributes=["case:private_insurance", "case:underlying_condition", "case:citizen", "case:german speaking", "case:gender"], possible_events=["Expert Examination"], to_remove=True))
        critical_decisions.append(Decision(attributes=["case:age"], possible_events=["Expert Examination"], to_remove=False))
    elif "cc_n" in folder_name:
        critical_decisions.append(Decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True))
        critical_decisions.append(Decision(attributes=["gender"], possible_events=["prostate screening", "mammary screening"], to_remove=False))
    
    return critical_decisions


# generates and or processes the data
def run_preprocessing(folder_name, file_name=None, model_name=None, n_gram=3, num_cases=1000):
    if model_name:
        X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices, critical_decisions = generate_data(num_cases, model_name, n_gram)
    elif file_name:
        df = load_xes_to_df(file_name, folder_name=folder_name)
        categorical_attributes, numerical_attributes = get_attributes(folder_name)
        X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)
    else:
        df = load_data(folder_name, "df.pkl")
        categorical_attributes, numerical_attributes = get_attributes(folder_name)
        X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)

    if folder_name:
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

    #"""
    for domain in domains:
        for degree in degrees:
            folder_name = f"{domain}_{degree}"
            file_name = f"{domain}_log_{degree}.xes"
            print(f"Loading XES data for: {folder_name}")
            load_xes_to_df(file_name, folder_name=folder_name)
            print(f"Processing data for: {folder_name}")
            run_preprocessing(folder_name=folder_name, file_name=file_name)
    #"""

    for domain in domains:
        for degree in degrees:
            folder_name = f"{domain}_{degree}"
            file_name = f"{domain}_log_{degree}.xes"
            print(f"Training for: {folder_name}")
            run_train_base(folder_name)


# executes the complete pipeline
def run_complete(folder_name, model_name=None, file_name=None, n_gram=3, num_cases=1000, preprocessing=False):
    if preprocessing:
        run_preprocessing(folder_name, model_name=model_name, file_name=file_name, n_gram=n_gram, num_cases=num_cases)
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
    run_modify(folder_name, node_ids=nodes_to_remove, console_output=False)
    run_finetuning(folder_name, console_output=False)
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

    print("finetuning mode: simple")
    nn = load_nn(folder_name, "nn.keras")
    y_train = nn.predict(X_train)
    nn_simple = finetune_nn(nn, X_train, y_modified, y_train=y_train, mode="simple", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_simple.keras")
    y_distilled = distill_nn(nn_simple, X_train)
    dt_simple = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_weighted.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("finetuning mode: changed")
    nn = load_nn(folder_name, "nn.keras")
    nn_changed = finetune_nn(nn, X_train, y_modified, y_train=y_train, mode="changed", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_changed.keras")
    y_distilled = distill_nn(nn_changed, X_train)
    dt_changed = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_changed.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("finetuning mode: weighted")
    nn = load_nn(folder_name, "nn.keras")
    nn_weighted = finetune_nn(nn, X_train, y_modified, y_train=y_train, mode="weighted", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_weighted.keras")
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


def run_modify(folder_name, node_ids=[], console_output=True):
    X_train  = load_data(folder_name, "X_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    dt_distilled = load_dt(folder_name, "dt.json")

    for node_id in node_ids:
        dt_distilled.delete_branch(node_id)
    save_dt(dt_distilled, folder_name, "dt_modified.json")
    if console_output:
        evaluate_dt(dt_distilled, X_test, y_test)

    y_modified = dt_distilled.predict(X_train)
    y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))
    save_data(y_encoded, folder_name, "y_modified.pkl")


def run_demo(folder_name="cc_n", n_gram=2, num_cases=1000, preprocessing=False):
    #run_transformer()
    run_unfair_data_preset()
    sys.exit()
    #df = load_xes_to_df("hospital_billing")
    #process_model = build_process_model(folder_name)
    #trace_generator = TraceGenerator(process_model=process_model)
    #generated_cases = trace_generator.generate_traces(start_time=datetime.now(), num_cases=num_cases)
    #df = cases_to_dataframe(generated_cases)
    folder_name = "hb_high"
    file_name = "hospital_log_high.xes"
    run_xes_preprocessing(file_name, folder_name=folder_name)
    df = load_data(folder_name, "df.pkl")
    print(df.head(20))
    X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = process_df(df, ["case:gender", "case:citizen", "case:private_insurance", "case:german speaking"], ["case:age"], folder_name="hb_high")
    nn = train_nn(X_train, y_train, folder_name=folder_name)
    y_distilled = distill_nn(nn, X_train)
    print("y_distilled")
    print(y_distilled.shape)
    dt_distilled = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("Base model:")
    evaluate_nn(nn, X_test, y_test)
    evaluate_dt(dt_distilled, X_test, y_test)

def run_transformer(folder_name):
    #df = load_xes_to_df("hospital_billing")
    #process_model = build_process_model(folder_name)
    #trace_generator = TraceGenerator(process_model=process_model)
    #generated_cases = trace_generator.generate_traces(start_time=datetime.now(), num_cases=num_cases)
    #df = cases_to_dataframe(generated_cases)
    #df = load_data(folder_name, "df.pkl")
    #print(df.head(20))
    df = load_data(folder_name, "df.pkl")
    print(df.head(20))
    padding_value = len(set(df['activity']))
    vocab_size = padding_value + 1
    categorical_attributes = []
    numerical_attributes = []
    # Split the DataFrame by case_id to ensure no case is split between train and test
    unique_case_ids = df['case_id'].unique()
    train_case_ids, test_case_ids = train_test_split(unique_case_ids, test_size=0.2, random_state=42)

    # Create train and test DataFrames
    train_df = df[df['case_id'].isin(train_case_ids)]
    test_df = df[df['case_id'].isin(test_case_ids)]

    # Process the training data
    input_sequences_train, target_sequences_train, categorical_data_train, numerical_data_train = process_data_padded(
        train_df, categorical_attributes, numerical_attributes
    )
    max_seq_len = input_sequences_train.shape[1]
    print(f"Max length: {max_seq_len}")

    print("First 20 input sequences:")
    print(input_sequences_train[:20])

    print("\nFirst 20 target sequences:")
    print(target_sequences_train[:20])

    print("\nFirst 20 categorical data entries:")
    for i, cat_data in enumerate(categorical_data_train[:20]):
        print(f"Categorical attribute {i}: {cat_data}")

    print("\nFirst 20 numerical data entries:")
    for i, num_data in enumerate(numerical_data_train[:20]):
        print(f"Numerical attribute {i}: {num_data}")

    # Process the test data
    input_sequences_test, target_sequences_test, categorical_data_test, numerical_data_test = process_data_padded(
        test_df, categorical_attributes, numerical_attributes, max_seq_len=max_seq_len
    )

    # Create the transformer model
    model = build_transformer_model(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        padding_value=padding_value,
        num_categorical=len(categorical_attributes),
        num_numerical=len(numerical_attributes),
        embed_dim=32,
        num_heads=2,
        ff_dim=64
    )

    # Train the model
    history = model.fit(
        [input_sequences_train] + categorical_data_train + numerical_data_train,
        target_sequences_train,
        batch_size=32,
        epochs=10,
        validation_split=0.2
    )

    # Evaluate the model
    results = model.evaluate(
        [input_sequences_test] + categorical_data_test + numerical_data_test,
        target_sequences_test
    )

    # Calculate and print accuracy
    print(f"Test Accuracy: {results[1] * 100:.2f}%")
    

def run_sklearn_test(folder_name="cc", n_gram=2, num_cases=10, preprocessing=False):
    if preprocessing:
        run_preprocessing(folder_name, n_gram=n_gram, num_cases=num_cases, console_output=False)
    nn = load_nn(folder_name, "nn.keras")
    X_train  = load_data(folder_name, "X_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")

    y_distilled = distill_nn(nn, X_train)
    start_time = time.time()
    dt_sklearn = train_sklearn_dt(X_train, y_distilled)
    end_time = time.time()
    print(f"Training time for sklearn decision tree: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    dt_distilled = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_distilled.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    end_time = time.time()
    print(f"Training time for distilled decision tree: {end_time - start_time:.2f} seconds")
    start_time = time.time()
    dt_transformed = sklearn_to_custom_tree(dt_sklearn, feature_names=feature_names, class_names=class_names, feature_indices=feature_indices)
    end_time = time.time()
    print(f"Training time for transformed decision tree: {end_time - start_time:.2f} seconds")

    print("Base NN:")
    evaluate_nn(nn, X_test, y_test)
    print("Directly distilled DT:")
    evaluate_dt(dt_distilled, X_test, y_test)
    print("Sklearn distilled DT:")
    evaluate_sklearn_dt(dt_sklearn, X_test, y_test, y_distilled, class_names=class_names, feature_names=feature_names)

    print("Transformed distilled DT:")
    evaluate_dt(dt_transformed, X_test, y_test)


def run_interactive():
    print("Python Interactive Shell. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            # Read user input
            command = input(">>> ")

            # Exit the shell if the user types 'exit' or 'quit'
            if command in ["exit", "quit"]:
                print("Exiting the shell.")
                break

            # Attempt to evaluate expressions (for things like math, variables, etc.)
            try:
                result = eval(command)
                if result is not None:
                    print(result)
            except SyntaxError:
                # If eval fails due to syntax error, it's likely a statement (e.g., assignment, import)
                exec(command)
        
        # Catch and display any runtime errors
        except Exception as e:
            print(f"Error: {e}")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting the shell.")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model to use (default: None)')
    parser.add_argument('--file_name', type=str, default=None, help='Name of the file to use (default: None)')
    parser.add_argument('--folder_name', type=str, default=None, help='Name of the folder to use (default: same as model_name)')
    parser.add_argument('--mode', choices=['a', 'c', 'd', 'f', 'i', 'm', 'p', 't'], default='c',
                        help="Choose 'complete' to run full pipeline or 'preprocessed' to use preprocessed data (default: complete)")
    parser.add_argument('--n_gram', type=int, default=3, help='Value for n-gram (default: 3)')
    parser.add_argument('--num_cases', type=int, default=1000, help='Number of cases to process (default: 1000)')
    parser.add_argument('--no-save', dest='save', action='store_false', 
                        help='Disable saving the results (default: results will be saved)')
    parser.add_argument('--p', dest='preprocessing', action='store_true', 
                        help='Generate new data')
    parser.add_argument("node_ids", type=int, nargs="*", help="List of node_ids")
    args = parser.parse_args()
    if args.folder_name is None:
        args.folder_name = args.model_name
    
    # Check which mode is selected and run the corresponding function
    if args.mode == 'a':
        run_analysis(args.folder_name)
    elif args.mode == 'c':
        run_complete(args.folder_name, model_name=args.model_name, file_name=args.file_name, n_gram=args.n_gram, num_cases=args.num_cases, preprocessing=args.preprocessing)
    elif args.mode == 'd':
        run_demo(args.folder_name, n_gram=args.n_gram, num_cases=args.num_cases, preprocessing=args.preprocessing)
    elif args.mode == 'f':
        run_finetuning(args.folder_name)
    elif args.mode == 'i':
        run_interactive()
    elif args.mode == 'm':
        run_modify(args.folder_name, node_ids=args.node_ids)
    elif args.mode == 'p':
        run_preprocessing(args.folder_name, model_name=args.model_name, file_name=args.file_name, n_gram=args.n_gram, num_cases=args.num_cases)
    elif args.mode == 't':
        run_train_base(args.folder_name)

if __name__ == "__main__":
    main()
    print("Done and dusted!")