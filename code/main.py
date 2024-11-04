import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score

from trace_generator import build_process_model
from data_processing import generate_processed_data, load_data, save_data
from decision_tree import DecisionTreeClassifier, save_tree_to_json, load_tree_from_json, get_deleted_nodes, get_metrics


def generate_data(num_cases, model_name, n_gram, save=True):
    process_model = build_process_model(model_name)
    folder_name = model_name if save else None
    X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = generate_processed_data(process_model, num_cases, n_gram, folder_name=folder_name)
    return X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices

# define neural network architecture
def build_nn(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
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
    print(f"input dimension: {input_dim}")
    print(f"output dimension: {output_dim}")
    print("--------------------------------------------------------------------------------------------------")
    print("training neural network:")
    model = build_nn(input_dim, output_dim)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    print("--------------------------------------------------------------------------------------------------")
    if folder_name:
        save_nn(model, folder_name, model_name)
    return model

def train_dt(X_train, y_train, folder_name=None, model_name=None, feature_names=None, feature_indices=None, class_names=None):
    print("training decision tree...")
    dt = DecisionTreeClassifier(class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    dt.fit(X_train, y_train)
    if model_name:
        save_dt(dt, folder_name, model_name)
    print("--------------------------------------------------------------------------------------------------")
    return dt

def save_nn(model, folder_name, file_name):
    print(f"saving {file_name}...")
    full_path = os.path.join('models', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    model.save(file_path)

def load_nn(folder_name, file_name):
    print(f"loading {file_name}...")
    file_name = os.path.join('models', folder_name, file_name)
    model = load_model(file_name)
    print("--------------------------------------------------------------------------------------------------")
    return model
    
def save_dt(dt, folder_name, file_name):
    print(f"saving {file_name}...")
    full_path = os.path.join('models', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    save_tree_to_json(dt, file_path)


def load_dt(folder_name, file_name):
    print(f"loading {file_name}...")
    file_path = os.path.join('models', folder_name, file_name)
    print("--------------------------------------------------------------------------------------------------")
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
        print(f"metrics for modified node {node_id}:")
        for metric in metrics:
            print(metric)


def evaluate_dt(dt, X_test, y_test):
    print("testing dt:")
    y_test = np.argmax(y_test, axis=1)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("")
    dt.visualize()
    print("")
    dt.print_tree_metrics(X_test, y_test, dt.root)
    print("--------------------------------------------------------------------------------------------------")

def distill_nn(nn, X, folder_name=None):
    print("distilling nn:")
    softmax_predictions = nn.predict(X)
    y = np.argmax(softmax_predictions, axis=1)
    if folder_name:
        save_data(y, folder_name, "y_distilled.pkl")
    print("--------------------------------------------------------------------------------------------------")
    return y

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

# executes the complete pipeline for the prerequisites
def run_complete(model_name, n_gram=2, num_cases=1000, save=True):
    X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = generate_data(num_cases, model_name, n_gram, save=save)
    folder_name = model_name if save else None
    nn = train_nn(X_train, y_train, folder_name=folder_name)

    y_distilled = distill_nn(nn, X_train, folder_name=folder_name)
    dt_distilled = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_distilled.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)

    evaluate_nn(nn, X_test, y_test)
    evaluate_dt(dt_distilled, X_test, y_test)


# executes the pipeline with preprocessed data and an already trained nn
def run_preprocessed(folder_name):
    X_train  = load_data(folder_name, "X_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    y_train  = load_data(folder_name, "y_train.pkl")
    y_distilled  = load_data(folder_name, "y_distilled.pkl")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")
    nn = load_nn(folder_name, "nn.keras")

    dt_distilled = train_dt(X_train, y_distilled, folder_name=folder_name, model_name="dt_distilled.json", class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)

    evaluate_nn(nn, X_test, y_test)
    evaluate_dt(dt_distilled, X_test, y_test)


# runs analysis on finished models
def run_analysis(folder_name):
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    dt_distilled = load_dt(folder_name, "dt_distilled.json")
    dt_modified = load_dt(folder_name, "dt_modified.json")
    folder_path = os.path.join('models', folder_name)
    modified_nodes = get_deleted_nodes(dt_distilled, dt_modified)

    # analyze all neural networks trees
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.keras'):
            print(f"Analyzing model: {file_name}")
            nn = load_nn(folder_name, file_name)
            evaluate_nn(nn, X_test, y_test)
            #print_metrics_nn(nn, dt_distilled, X_test, node_ids=modified_nodes)

    # analyze all decision trees
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            print(f"Analyzing model: {file_name}")
            dt = load_dt(folder_name, file_name)
            evaluate_dt(dt, X_test, y_test)


# runs finetuning for modified base data
def run_finetuning(folder_name="model_1", epochs=5, batch_size=32, learning_rate=1e-3, weight=5):
    X_train  = load_data(folder_name, "X_train.pkl")
    y_train  = load_data(folder_name, "y_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    y_modified = load_data(folder_name, "y_modified.pkl")
    y_distilled = load_data(folder_name, "y_distilled.pkl")
    dt_distilled = load_dt(folder_name, "dt_distilled.json")
    dt_modified = load_dt(folder_name, "dt_modified.json")
    class_names = load_data(folder_name, "class_names.pkl")
    feature_names = load_data(folder_name, "feature_names.pkl")
    feature_indices = load_data(folder_name, "feature_indices.pkl")

    print("finetuning mode: simple")
    nn = load_nn(folder_name, "nn.keras")
    y_train = nn.predict(X_train)
    nn_simple = finetune_nn(nn, X_train, y_modified, y_train=y_train, mode="simple", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_simple.keras")
    y_distilled = distill_nn(nn_simple, X_train, folder_name=folder_name)
    dt_simple = train_dt(X_train, y_distilled, folder_name=folder_name, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("finetuning mode: changed")
    nn = load_nn(folder_name, "nn.keras")
    nn_changed = finetune_nn(nn, X_train, y_modified, y_train=y_train, mode="changed", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_changed.keras")
    y_distilled = distill_nn(nn_changed, X_train, folder_name=folder_name)
    dt_changed = train_dt(X_train, y_distilled, folder_name=folder_name, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)
    print("finetuning mode: weighted")
    nn = load_nn(folder_name, "nn.keras")
    nn_weighted = finetune_nn(nn, X_train, y_modified, y_train=y_train, mode="weighted", epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, weight=weight)
    save_nn(nn_simple, folder_name, "nn_weighted.keras")
    y_distilled = distill_nn(nn_weighted, X_train, folder_name=folder_name)
    dt_weighted = train_dt(X_train, y_distilled, folder_name=folder_name, class_names=class_names, feature_names=feature_names, feature_indices=feature_indices)

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


def run_modify(folder_name="model_1", node_ids=[], save=True):
    X_train  = load_data(folder_name, "X_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    dt_distilled = load_dt(folder_name, "dt_distilled.json")

    for node_id in node_ids:
        dt_distilled.delete_branch(node_id)
    save_dt(dt_distilled, folder_name, "dt_modified.json")
    evaluate_dt(dt_distilled, X_test, y_test)

    y_modified = dt_distilled.predict(X_train)
    y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))
    if save:
        save_data(y_encoded, folder_name, "y_modified.pkl")


def run_demo(folder_name="model_1", n_gram=2, num_cases=1000, save=False):
    X_train  = load_data(folder_name, "X_train.pkl")
    y_train  = load_data(folder_name, "y_train.pkl")
    X_test  = load_data(folder_name, "X_test.pkl")
    y_test  = load_data(folder_name, "y_test.pkl")
    nn = load_nn(folder_name, "nn.keras")
    dt_distilled = load_dt(folder_name, "dt_distilled.json")

    evaluate_nn(nn, X_test, y_test)
    evaluate_dt(dt_distilled, X_test, y_test)

    dt_distilled.delete_branch(9)
    save_dt(dt_distilled, folder_name, "dt_modified.json")
    y_modified = dt_distilled.predict(X_train)
    y_encoded = to_categorical(y_modified, num_classes=len(dt_distilled.class_names))
    save_data(y_encoded, folder_name, "y_modified.pkl")
    evaluate_dt(dt_distilled, X_test, y_test)


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
    parser.add_argument('--model_name', type=str, default='model_1', help='Name of the model to use (default: model_1)')
    parser.add_argument('--mode', choices=['a', 'c', 'd', 'f', 'i', 'm', 'p'], default='c',
                        help="Choose 'complete' to run full pipeline or 'preprocessed' to use preprocessed data (default: complete)")
    parser.add_argument('--n_gram', type=int, default=3, help='Value for n-gram (default: 3)')
    parser.add_argument('--num_cases', type=int, default=1000, help='Number of cases to process (default: 1000)')
    parser.add_argument('--no-save', dest='save', action='store_false', 
                        help='Disable saving the results (default: results will be saved)')
    args = parser.parse_args()
    
    # Check which mode is selected and run the corresponding function
    if args.mode == 'a':
        run_analysis(folder_name=args.model_name)
    elif args.mode == 'c':
        run_complete(model_name=args.model_name, n_gram=args.n_gram, num_cases=args.num_cases, save=args.save)
    elif args.mode == 'd':
        run_demo(folder_name=args.model_name, n_gram=args.n_gram, num_cases=args.num_cases, save=args.save)
    elif args.mode == 'f':
        run_finetuning(folder_name=args.model_name)
    elif args.mode == 'i':
        run_interactive()
    elif args.mode == 'm':
        run_modify(folder_name=args.model_name, node_ids=args.node_ids)
    elif args.mode == 'p':
        run_preprocessed(folder_name=args.model_name)

if __name__ == "__main__":
    main()
    print("Done and dusted!")
    print("new commit")