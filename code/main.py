import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import time
import sys
import shap
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
    X, y, class_names, feature_names, feature_indices, critical_decisions = generate_processed_data(process_model, categorical_attributes=categorical_attributes, numerical_attributes=numerical_attributes, num_cases=num_cases, n_gram=n_gram, folder_name=folder_name)
    return X, y, class_names, feature_names, feature_indices, critical_decisions

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
    # Embedding layer with mask_zero=True creates a mask
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    activity_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(activity_input)

    # Extract the mask from the embedding layer
    mask = activity_embedding._keras_mask  # Keras propagates this mask automatically

    # Positional Encoding
    position_input = tf.range(start=0, limit=max_seq_len, delta=1)
    position_embedding = Embedding(input_dim=max_seq_len, output_dim=embed_dim)(position_input)
    x = activity_embedding + position_embedding

    # Transformer Encoder Block
    mask = ExpandDimsLayer(axis=1)(mask)  # Adjust dimensions for MultiHeadAttention

    # Transformer Encoder Block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        activity_embedding, activity_embedding, attention_mask=mask
    )
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation='relu')(attn_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(embed_dim)(ff_output)
    x = LayerNormalization(epsilon=1e-6)(attn_output + ff_output)

    # Concatenate additional features
    categorical_inputs, numerical_inputs = [], []
    if num_categorical > 0:
        # Create inputs and embeddings for categorical features
        cat_inputs = [Input(shape=(max_seq_len,), name=f"cat_input_{i}") for i in range(num_categorical)]
        cat_embeddings = [Embedding(100, embed_dim)(cat_input) for cat_input in cat_inputs]
        x = Concatenate(axis=-1)([x] + cat_embeddings)
        categorical_inputs.extend(cat_inputs)

    # Handle numerical inputs
    if num_numerical > 0:
        # Create inputs with shape (max_seq_len, 1) to include time dimension
        num_inputs = [Input(shape=(max_seq_len, 1), name=f"num_input_{i}") for i in range(num_numerical)]
        num_dense = [Dense(embed_dim)(num_input) for num_input in num_inputs]
        x = Concatenate(axis=-1)([x] + num_dense)
        numerical_inputs.extend(num_inputs)

    # Output layer
    output = Dense(vocab_size, activation='softmax', name='output')(x)

    # Define model
    inputs = [activity_input] + categorical_inputs + numerical_inputs
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
    #dt = SklearnDecisionTreeClassifier(max_depth=10)
    #dt = SklearnDecisionTreeClassifier(max_depth=10, max_leaf_nodes=50, min_samples_leaf=5)
    dt = SklearnDecisionTreeClassifier(max_depth=10, max_leaf_nodes=50, min_samples_leaf=5, ccp_alpha=0.001)
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

def calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X, critical_decisions, feature_indices, class_names, feature_names, base_attributes):
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
                        protected_attribute = X[:, feature_index]
                        stat_par, disp_imp = get_fairness_metrics(y[name], protected_attribute, event_index)

                        disp_imp_results[outer_key][name] = disp_imp
                        stat_par_results[outer_key][name] = stat_par
                        print(f"Disparate Impact for {feature_names[feature_index]}, {event}, {name}: {disp_imp}")
                        print(f"Statistical Parity for {feature_names[feature_index]}, {event}, {name}: {stat_par}")

    return stat_par_results, disp_imp_results

def remove_attribute_features(X, feature_indices, base_attributes):
    remove_indices = [idx for attr, indices in feature_indices.items() if attr not in base_attributes for idx in indices]
    return np.delete(X, remove_indices, axis=1)

def get_fairness_metrics(y, protected_attribute, event_index):
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    y_binary = np.zeros_like(y)
    y_binary[y == event_index] = 1
    total_amount = np.sum(y_binary == 1)
    if total_amount < 1:
        return 0, 1

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

    stat_parity = selection_rate_unprivileged - selection_rate_privileged
    return stat_parity, disp_impact


def calculate_shapley_scores(model, X, relevant_attributes, feature_names, output_index, background_size=1000):
    # Verify that relevant attributes are in feature_names
    assert all(attr in feature_names for attr in relevant_attributes), "Relevant attributes must exist in feature names."

    # Map relevant attributes to their indices in X
    relevant_indices = [feature_names.index(attr) for attr in relevant_attributes]
    print(relevant_indices)

    if X.shape[0] > background_size:
        X = X[np.random.choice(X.shape[0], size=background_size, replace=False), :]

    # Initialize SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, X)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)  # Returns a list of arrays (one for each output class)
    print(shap_values.shape)

    # For simplicity, aggregate shap values across all classes (sum of absolute values)
    #shap_values = np.sum(np.abs(np.array(shap_values)), axis=0)
    shap_values = shap_values[:, :, output_index]
    print(shap_values.shape)

    shapley_scores = {}
    for i, idx in enumerate(relevant_indices):
        shapley_scores[relevant_attributes[i]] = np.mean(np.abs(shap_values[:, idx]))
        if i in [34, 35]:
            plot_density(shap_values[:, idx], idx)
    print(shapley_scores)

    return shapley_scores

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


# generates and or processes the data
def run_preprocessing(folder_name, file_name=None, model_name=None, n_gram=3, num_cases=10000, enrichment=False):
    if model_name:
        print(model_name)
        X, y, class_names, feature_names, feature_indices, critical_decisions = generate_data(num_cases, model_name, n_gram)
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
def run_complete(folder_name, model_name=None, file_name=None, n_gram=3, num_cases=1000, preprocessing=False, train_base=False, enrichment=False, retrain=False):
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
    if nodes_to_remove:
        print(f"Nodes to remove accordingly: {nodes_to_remove}")
        print("--------------------------------------------------------------------------------------------------")
        run_modify(folder_name, node_ids=nodes_to_remove, console_output=False, retrain=retrain)
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

def run_demo(folder_name, n_gram=3, num_cases=1000, preprocessing=False):
    run_shapley("bpi_A")
    sys.exit()
    male_shapley_scores = load_data(folder_name, "male_shapley_scores.pkl")
    female_shapley_scores = load_data(folder_name, "female_shapley_scores.pkl")
    male_shapley_scores_modified = load_data(folder_name, "male_shapley_scores_modified.pkl")
    female_shapley_scores_modified = load_data(folder_name, "female_shapley_scores_modified.pkl")
    plot_shapley(male_shapley_scores, male_shapley_scores_modified, folder_name, "shapley_male.png")
    plot_shapley(female_shapley_scores, female_shapley_scores_modified, folder_name, "shapley_female.png")

def run_shapley(folder_name, n_gram=3):
    enriched_accuracy_values = []
    modified_accuracy_values = []
    male_shapley_scores = []
    female_shapley_scores = []
    male_shapley_scores_modified = []
    female_shapley_scores_modified = []
    df = load_data(folder_name, "df.pkl")
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    critical_decisions = get_critical_decisions(folder_name)
    X_enriched, y_enriched, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)
    if isinstance(class_names, list):
        output_index = class_names.index("A_DECLINED")
    elif isinstance(class_names, np.ndarray):
        output_index = np.where(class_names == "A_DECLINED")[0][0]


    for _ in tqdm(range(3), desc="evaluating model:"):
        # evaluating the enriched model
        X_train, X_test, y_train, y_test = train_test_split(X_enriched, y_enriched, test_size=0.3)
        nn = train_nn(X_train, y_train)
        y_distilled = distill_nn(nn, X_train)
        dt_distilled = train_dt(X_train, y_distilled, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
        enriched_accuracy = evaluate_nn(nn, X_test, y_test)
        enriched_accuracy_values.append(enriched_accuracy)
        shapley_scores = calculate_shapley_scores(nn, X_test, feature_names, feature_names, output_index)
        male_shapley_score = shapley_scores["gender = male"]
        female_shapley_score = shapley_scores["gender = female"]
        male_shapley_scores.append(male_shapley_score)
        female_shapley_scores.append(female_shapley_score)

        # modifying the distilled model
        nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
        if nodes_to_remove:
            print(f"Removing nodes: {nodes_to_remove}")
            y_encoded = np.argmax(y_train, axis=1)
            for node_id in nodes_to_remove:
                dt_distilled.delete_node(X_train, y_encoded, node_id)
            y_modified = dt_distilled.predict(X_train)
            y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))

            # finetuning and evaluating the changed model
            nn_modified = finetune_nn(nn, X_train, y_encoded, y_train=y_train, mode="changed")
            modified_accuracy = evaluate_nn(nn_modified, X_test, y_test)
            shapley_scores = calculate_shapley_scores(nn_modified, X_test, feature_names, feature_names, output_index)
            male_shapley_score_modified = shapley_scores["gender = male"]
            female_shapley_score_modified = shapley_scores["gender = female"]
        else:
            print(f"No nodes to remove!")
            male_shapley_score_modified = male_shapley_score
            female_shapley_score_modified = female_shapley_score
            modified_accuracy = enriched_accuracy

        male_shapley_scores_modified.append(male_shapley_score_modified)
        female_shapley_scores_modified.append(female_shapley_score_modified)
        modified_accuracy_values.append(modified_accuracy)

        print(f"enriched accuracy: {enriched_accuracy}, modified accuracy: {modified_accuracy}")
        print(f"enriched: male shapley score: {male_shapley_score}, female shapley score: {female_shapley_score}")
        print(f"modified: male shapley score: {male_shapley_score_modified}, female shapley score: {female_shapley_score_modified}")
        print("--------------------------------------------------------------------------------------------------")

    save_data(male_shapley_scores, folder_name, "male_shapley_scores.pkl")
    save_data(female_shapley_scores, folder_name, "female_shapley_scores.pkl")
    save_data(male_shapley_scores_modified, folder_name, "male_shapley_scores_modified.pkl")
    save_data(female_shapley_scores_modified, folder_name, "female_shapley_scores_modified.pkl")
    save_data(enriched_accuracy_values, folder_name, "enriched_accuracy_values.pkl")
    save_data(modified_accuracy_values, folder_name, "modified_accuracy_values.pkl")
    plot_shapley(male_shapley_scores, male_shapley_scores_modified, folder_name, "shapley_male.png")
    plot_shapley(female_shapley_scores, female_shapley_scores_modified, folder_name, "shapley_female.png")



def run_4_cases():
    #folder_names = ["hb_-age_-gender", "hb_-age_+gender", "hb_+age_-gender", "hb_+age_+gender"]
    folder_names = ["hb_-age_+gender"]

    for folder_name in folder_names:
        load_xes_to_df("hospital_billing", folder_name=folder_name)
        run_enrichment(folder_name)
        run_evaluate(folder_name)


def run_enrichment(folder_name):
    print("Running enrichment...")
    df = load_data(folder_name, "df.pkl")
    rules = get_rules(folder_name)
    df = enrich_df(df, rules, folder_name)
    print(df.head(20))
    save_data(df, folder_name, "df.pkl")

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
    categorical_attributes = ["gender"]
    numerical_attributes = ["age"]
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

def run_evaluate(folder_name, iterations=10, n_gram=3, retrain=False):
    base_accuracy_values = []
    enriched_accuracy_values = []
    modified_accuracy_values = []
    disp_imp_results = {}
    stat_par_results = {}
    df = load_data(folder_name, "df.pkl")
    categorical_attributes, numerical_attributes = get_attributes(folder_name)
    categorical_attributes_base, numerical_attributes_base = [], ["case:AMOUNT_REQ"]
    base_attributes = categorical_attributes_base + numerical_attributes_base
    critical_decisions = get_critical_decisions(folder_name)
    #X_base, y_base, class_names, feature_names, feature_indices = process_df(df, categorical_attributes_base, numerical_attributes_base, n_gram=n_gram)
    X, y, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram)

    for i in tqdm(range(iterations), desc="evaluating model:"):
        # evaluating the base model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_train_base = remove_attribute_features(X_train, feature_indices, base_attributes)
        X_test_base = remove_attribute_features(X_test, feature_indices, base_attributes)
        nn_base = train_nn(X_train_base, y_train)
        base_accuracy = evaluate_nn(nn_base, X_test_base, y_test)
        base_accuracy_values.append(base_accuracy)

        # evaluating the enriched model
        nn_enriched = train_nn(X_train, y_train)
        y_distilled = distill_nn(nn_enriched, X_train)
        dt_distilled = train_dt(X_train, y_distilled, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
        enriched_accuracy = evaluate_nn(nn_enriched, X_test, y_test)
        evaluate_dt(dt_distilled, X_test, y_test)
        enriched_accuracy_values.append(enriched_accuracy)

        # modifying the distilled model
        nodes_to_remove = dt_distilled.find_nodes_to_remove(critical_decisions)
        if nodes_to_remove:
            print(f"Removing nodes: {nodes_to_remove}")
            if retrain:
                y_encoded = np.argmax(y_train, axis=1)
                for node_id in nodes_to_remove:
                    dt_distilled.delete_node(X_train, y_encoded, node_id)
            else:
                for node_id in nodes_to_remove:
                    dt_distilled.delete_branch(node_id)
            y_modified = dt_distilled.predict(X_train)
            y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))

            # finetuning and evaluating the changed model
            nn_modified = clone_model(nn_enriched)
            nn_modified.set_weights(nn_enriched.get_weights())
            nn_modified = finetune_nn(nn_modified, X_train, y_encoded, y_train=y_train, mode="changed")
            modified_accuracy = evaluate_nn(nn_modified, X_test, y_test)
        else:
            print(f"No nodes to remove!")
            modified_accuracy = enriched_accuracy
        modified_accuracy_values.append(modified_accuracy)

        print(f"base accuracy: {base_accuracy}, enriched accuracy: {enriched_accuracy}, modified accuracy: {modified_accuracy}")
        stat_par_result, disp_imp_result = calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes)
        for outer_key, outer_value in stat_par_result.items():
            if i == 0:
                stat_par_results[outer_key] = {}
            for inner_key, inner_value in outer_value.items():
                if i == 0:
                    stat_par_results[outer_key][inner_key] = []
                stat_par_results[outer_key][inner_key].append(inner_value)
        for outer_key, outer_value in disp_imp_result.items():
            if i == 0:
                disp_imp_results[outer_key] = {}
            for inner_key, inner_value in outer_value.items():
                if i == 0:
                    disp_imp_results[outer_key][inner_key] = []
                disp_imp_results[outer_key][inner_key].append(inner_value)
        print("--------------------------------------------------------------------------------------------------")

    save_data(base_accuracy_values, folder_name, "base_accuracy_values.pkl")
    save_data(enriched_accuracy_values, folder_name, "enriched_accuracy_values.pkl")
    save_data(modified_accuracy_values, folder_name, "modified_accuracy_values.pkl")
    plot_accuracy(base_accuracy_values, enriched_accuracy_values, modified_accuracy_values, folder_name)
    plot_all_fairness(stat_par_results, disp_imp_results, folder_name)
    return base_accuracy_values, modified_accuracy_values

def run_fairness(folder_name):
    nn_base = load_nn(folder_name, "nn_base.keras")
    nn_enriched = load_nn(folder_name, "nn_enriched.keras")
    nn_modified = load_nn(folder_name, "nn_modified.keras")
    X_test  = load_data(folder_name, "X_test.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")
    critical_decisions = get_critical_decisions(folder_name)
    base_attributes = ["case:AMOUNT_REQ"]
    disp_imp_results = {}

    disp_imp_result = calculate_comparable_fairness(nn_base, nn_enriched, nn_modified, X_test, critical_decisions, feature_indices, class_names, feature_names, base_attributes)
    print(disp_imp_result)
    for outer_key, outer_value in disp_imp_result.items():
        disp_imp_results[outer_key] = {}
        for inner_key, inner_value in outer_value.items():
            disp_imp_results[outer_key][inner_key] = []
            disp_imp_results[outer_key][inner_key].append(inner_value)
    print(disp_imp_results)
    #plot_all_fairness(stat_par_results, disp_imp_results, folder_name)


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
    parser.add_argument('--mode', choices=['a', 'c', 'd', 'e', 'f', 'i', 'm', 'p', 't', 'v'], default='c',
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
    elif args.mode == 'm':
        run_modify(args.folder_name, node_ids=args.node_ids, retrain=args.retrain)
    elif args.mode == 'p':
        run_preprocessing(args.folder_name, model_name=args.model_name, file_name=args.file_name, n_gram=args.n_gram, num_cases=args.num_cases)
    elif args.mode == 't':
        run_train_base(args.folder_name)
    elif args.mode == 'v':
        run_evaluate(args.folder_name, retrain=args.retrain)

if __name__ == "__main__":
    main()
    print("Done and dusted!")