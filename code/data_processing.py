import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pm4py

from typing import List, Dict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from trace_generator import Event, Case, TraceGenerator
from tqdm import tqdm 

from main import print_samples

from scipy.stats import norm, truncnorm
rng = np.random.default_rng(0)

def load_data(folder_name, file_name):
    file_path = os.path.join('processed_data', folder_name, file_name)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def save_data(data, folder_name, file_name):
    full_path = os.path.join('processed_data', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_csv_to_df(file_name):
    file_path = os.path.join('raw_data', file_name)
    return pd.read_csv(file_path)

def load_xes_to_df(file_name, folder_name=None, num_cases=10000):
    file_path = os.path.join("raw_data", file_name)
    df = pm4py.read_xes(file_path)
    df.rename(columns={'case:concept:name': 'case_id', 'concept:name': 'activity'}, inplace=True)
    columns = ['case_id', 'activity'] + [col for col in df.columns if col not in ['case_id', 'activity']]
    df = df[columns]
    df = df[df['case_id'].isin(df['case_id'].unique()[:num_cases])]
    df = df.loc[:, ~df.columns.duplicated()]
    df = process_df_timestamps(df)
    if folder_name:
        save_data(df, folder_name, "df.pkl")
    pd.set_option('display.max_columns', None)  # Display all columns
    print(df.head(20))
    return df

def generate_processed_data(process_model, categorical_attributes=[], numerical_attributes=[], num_cases=1000, n_gram=3, folder_name=None):
    print("generating event traces:")
    trace_generator = TraceGenerator(process_model=process_model)
    cases = trace_generator.generate_traces(num_cases=num_cases)
    print("example trace:")
    print(cases[:1])
    print("event pool:")
    print(trace_generator.get_events())
    print("--------------------------------------------------------------------------------------------------")

    print("processing nn data:")
    df = cases_to_dataframe(cases)
    X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = process_df(df, categorical_attributes, numerical_attributes, n_gram=n_gram, folder_name=folder_name)
    print("--------------------------------------------------------------------------------------------------")
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
        save_data(df, folder_name, "df.pkl")
        save_data(process_model.critical_decisions, folder_name, "critical_decisions.pkl")

    return X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices, process_model.critical_decisions

def create_feature_names(event_pool, attribute_pools, numerical_attributes, n_gram):
    feature_names = []
    # Add event features for each n-gram step
    for index in range(n_gram, 0, -1):
        for event in sorted(event_pool):
            feature_names.append(f"-{index}. Event = {event}")
    # Add categorical attribute features
    for attribute_name, possible_values in sorted(attribute_pools.items()):
        for value in sorted(possible_values):
            feature_names.append(f"{attribute_name} = {value}")
    # Add numerical attribute features
    for numerical_attr in sorted(numerical_attributes):
        feature_names.append(numerical_attr)
    return feature_names

def create_feature_indices(event_pool, attribute_pools, numerical_attributes, n_gram):
    num_events = len(sorted(event_pool))  # Sorted event pool
    feature_indices = {}
    # Allocate indices for events
    index = num_events * n_gram
    # Allocate indices for categorical attributes
    for attribute_name, possible_values in sorted(attribute_pools.items()):
        num_values = len(sorted(possible_values))
        feature_indices[attribute_name] = list(range(index, index + num_values))
        index += num_values
    # Allocate indices for numerical attributes
    for numerical_attr in sorted(numerical_attributes):
        feature_indices[numerical_attr] = [index]
        index += 1
    return feature_indices
    
def create_attribute_pools(df, case_attributes):
    attribute_pools = {}
    for attr in case_attributes:
        if attr in df.columns:
            attribute_pools[attr] = sorted(df[attr].dropna().unique().tolist())  # Ensure sorted order
        else:
            raise KeyError(f"Attribute '{attr}' is not in the DataFrame.")
    return attribute_pools 

def process_df_timestamps(df):
    if "time" in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by=['case_id', 'time']).reset_index(drop=True)
        df['time_delta'] = df.groupby('case_id')['time'].diff().dt.total_seconds()
        df['time_delta'] = df['time_delta'].fillna(0)  # Set time_delta to 0 for the first event in each case
        df['time_of_day'] = df['time'].dt.hour / 24 + df['time'].dt.minute / 1440 + df['time'].dt.second / 86400
        df['day_of_week'] = df['time'].dt.dayofweek  # Monday=0, Sunday=6
    return df

def cases_to_dataframe(cases: List[Case]) -> pd.DataFrame:
    """
    Converts a list of Case objects into a pandas DataFrame with columns:
    'case_id', 'activity', and one column for each case attribute (categorical and numerical).
    """
    rows = []
    for case in cases:
        for event in case.events:
            row = {
                'case_id': case.case_id,
                'activity': event.activity,
                'time': event.timestamp
            }
            row.update(case.categorical_attributes)
            row.update(case.numerical_attributes)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = process_df_timestamps(df)
    return df

def process_data_padded(df, categorical_attributes, numerical_attributes, max_seq_len=None):
    """
    Transforms a dataframe into sequences for transformer model training.

    Args:
        df (pd.DataFrame): DataFrame with columns case_id, activity, timestamp, and attributes.
        categorical_attributes (list): List of categorical attribute column names.
        numerical_attributes (list): List of numerical attribute column names.
        max_seq_len (int, optional): Max sequence length for padding. Default is max sequence length in the data.

    Returns:
        tuple: Processed input sequences, target sequences, categorical, and numerical metadata.
    """
    #df = df.sort_values(by=["case_id", "timestamp"])
    padding_value = 0
    le_activity = LabelEncoder()
    df.loc[:, 'activity_encoded'] = le_activity.fit_transform(df['activity']) + 1

    grouped = df.groupby("case_id")
    activity_sequences = grouped["activity_encoded"].apply(list).tolist()
    
    # Prepare input and target sequences
    input_sequences = [seq[:-1] for seq in activity_sequences]
    target_sequences = [seq[1:] for seq in activity_sequences]

    if max_seq_len is None:
        max_seq_len = max(len(seq) for seq in input_sequences)

    if padding_value is None:
        padding_value = max_seq_len

    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre', value=padding_value)
    target_sequences = pad_sequences(target_sequences, maxlen=max_seq_len, padding='pre', value=padding_value)

    # Encode categorical attributes
    categorical_data = []
    if len(categorical_attributes) > 0:
        for attr in categorical_attributes:
            le = LabelEncoder()
            df.loc[:, attr] = le.fit_transform(df[attr])
            grouped_attr = grouped[attr].apply(list)
            categorical_data.append(
                pad_sequences(grouped_attr.apply(lambda x: x[:max_seq_len]), maxlen=max_seq_len, padding='pre', value=padding_value)
            )

    # Scale numerical attributes
    numerical_data = []
    if len(numerical_attributes) > 0:
        scaler = MinMaxScaler()
        df.loc[:, numerical_attributes] = scaler.fit_transform(df[numerical_attributes])
        for attr in numerical_attributes:
            grouped_attr = grouped[attr].apply(list)
            numerical_data.append(
                pad_sequences(grouped_attr.apply(lambda x: x[:max_seq_len]), maxlen=max_seq_len, padding='pre', value=padding_value, dtype='float32')
            )

    return input_sequences, target_sequences, categorical_data, numerical_data


#TODO: leaking test info for minmax
def process_df(df, categorical_attributes, numerical_attributes, n_gram=3, folder_name=None, test_size=0.2):
    """Processes dataframe data for neural network training"""
    # keep only specified attributes
    standard_attributes = ['case_id', 'activity']
    categorical_attributes.sort()
    numerical_attributes.sort()
    attributes_to_include = standard_attributes + categorical_attributes + numerical_attributes 
    print(attributes_to_include)
    df = df[attributes_to_include]

    # create the meta-information
    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, categorical_attributes)
    feature_names = create_feature_names(class_names, attribute_pools, numerical_attributes, n_gram)
    feature_indices = create_feature_indices(class_names, attribute_pools, numerical_attributes, n_gram)
    print(f"class_names: {class_names}")
    print(f"amount of classes = {len(class_names)}, ngram = {n_gram}")
    print(f"attribute_pools: {attribute_pools}")
    print(f"feature_names: {feature_names}")
    print(f"feature_indices: {feature_indices}")

    # one-hot encode activities
    activity_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[class_names])
    activity_ohe = activity_encoder.fit_transform(df[['activity']])
    pad_activity_idx = activity_encoder.categories_[0].tolist().index("<PAD>")
    
    # one-hot encode categorical case attributes dynamically
    attribute_encoders = {}
    attributes_ohe_dict = {}
    for attr in categorical_attributes:
        print(attr)
        print(attribute_pools[attr])
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[attribute_pools[attr]])
        attributes_ohe_dict[attr] = encoder.fit_transform(df[[attr]])
        attribute_encoders[attr] = encoder

    # Scale numerical attributes between 0 and 1 based on training data's min/max values
    numerical_scalers = {}
    for attr in numerical_attributes:
        scaler = MinMaxScaler()
        scaler.fit(df[[attr]])
        numerical_scalers[attr] = scaler

    # group by case_id and create sequences
    grouped = df.groupby('case_id')
    cases = []
    for case_id, group in tqdm(grouped, desc="preparing cases"):
        activities = activity_encoder.transform(group[['activity']])
        attributes = {attr: attribute_encoders[attr].transform(group[[attr]]) for attr in categorical_attributes}
        # Scale numerical attributes within the group
        for attr in numerical_attributes:
            group[attr] = numerical_scalers[attr].transform(group[[attr]])
        cases.append((activities, attributes, group[numerical_attributes].values))
    pd.set_option('display.max_columns', None)  # Display all columns
    print(grouped.head(20))

    # Generate n-grams with padding
    X, y = [], []
    pad_activity = activity_encoder.transform([["<PAD>"]])  # Get one-hot for <PAD>
    pad_attributes = {attr: np.zeros((1, attributes_ohe_dict[attr].shape[1])) for attr in categorical_attributes}
    pad_numerical = np.zeros((1, len(numerical_attributes)))  # Padding for numerical features

    for activities, attributes, numerical in tqdm(cases, desc="encoding cases"):
        # Pad activities
        padded_activities = np.vstack([pad_activity] * n_gram + [activities])
        
        # Pad categorical attributes
        padded_attributes = {attr: np.vstack([pad_attributes[attr]] * n_gram + [attributes[attr]])
                            for attr in sorted(attributes)}

        # Pad numerical attributes
        padded_numerical = np.vstack([pad_numerical] * n_gram + [numerical])

        for i in range(len(activities)):  # Start from 0 and include all real activities
            x_activities = padded_activities[i:i + n_gram]
            if categorical_attributes:
                x_attributes = np.hstack([
                    padded_attributes[attr][i + n_gram] 
                    for attr in categorical_attributes
                ])
            else:
                x_attributes = np.array([])

            if numerical_attributes:
                x_numerical = padded_numerical[i + n_gram]
            else:
                x_numerical = np.array([])

            x_combined = np.hstack([x_activities.flatten(), x_attributes, x_numerical])  # Combine activities, attributes, and numerical features
            
            y_next_activity = activities[i]  # Predict the actual next activity
            X.append(x_combined)
            y.append(y_next_activity)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    n = 10
    print("example nn inputs:")
    print(X[:n])
    print_samples(n, X, y, class_names, feature_names, numerical_attributes)
    print("example nn outputs:")
    print(y[:n])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices

def enrich_df(df: pd.DataFrame, rules: list, folder_name: str):
    def generate_value(distribution, rng):
        """Generates a value based on the specified distribution."""
        if distribution['type'] == 'discrete':
            values, weights = zip(*distribution['values'])
            return rng.choice(values, p=np.array(weights) / sum(weights))
        elif distribution['type'] == 'normal':
            mean, std = distribution['mean'], distribution['std']
            a, b = distribution.get('min', -np.inf), distribution.get('max', np.inf)

            if np.isinf(a) and np.isinf(b):
                return norm.rvs(loc=mean, scale=std, random_state=rng)
            else:
                # Scale the bounds relative to the mean and standard deviation
                a, b = (a - mean) / std, (b - mean) / std
                return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)
        else:
            raise ValueError("Unsupported distribution type.")

    rng = np.random.default_rng(0)

    # Identify unique cases in the log
    case_ids = df['case_id'].unique()

    # Initialize a dictionary to hold the generated attributes for each case
    case_attributes = {rule['attribute']: {} for rule in rules}

    for case_id in tqdm(case_ids, desc="enriching cases"):
        # Extract events for the current case
        case_events = df[df['case_id'] == case_id]['activity'].tolist()

        for rule in rules:
            subsequence = rule['subsequence']
            attribute = rule['attribute']
            distribution = rule['distribution']

            # Check if the subsequence is present in the case's events
            if any(
                case_events[i:i + len(subsequence)] == subsequence
                for i in range(len(case_events) - len(subsequence) + 1)
            ):
                # Generate a value based on the distribution
                case_attributes[attribute][case_id] = generate_value(distribution, rng)
                #print(f"Matched: {subsequence} to {case_attributes[attribute][case_id]}")

    # Add generated attributes as new columns to the DataFrame
    for attribute, values in case_attributes.items():
        df[attribute] = df['case_id'].map(values)
    
    # plot these for evaluation of the success
    plot_attributes(df, rules, folder_name)

    return df

def plot_attributes(df: pd.DataFrame, rules: list, folder_name: str):

    img_folder = f"img/{folder_name}"
    os.makedirs(img_folder, exist_ok=True)
    # Group by case_id to ensure each case is only counted once
    grouped = df.groupby('case_id')

    # Collect unique attributes and their rules
    attribute_rules = {}
    for rule in rules:
        attribute = rule['attribute']
        if attribute not in attribute_rules:
            attribute_rules[attribute] = []
        attribute_rules[attribute].append(rule)

    for attribute, rules in attribute_rules.items():
        # Combine data for the attribute
        attribute_values = grouped[attribute].first().dropna()

        # Start plotting
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Handle discrete attributes
        if any(rule['distribution']['type'] == 'discrete' for rule in rules):
            # Discrete values and their labels
            discrete_values = []
            for rule in rules:
                if rule['distribution']['type'] == 'discrete':
                    values, _ = zip(*rule['distribution']['values'])
                    discrete_values.extend(values)
            discrete_values = list(set(discrete_values))  # Remove duplicates

            # Calculate percentages
            counts = attribute_values.value_counts(normalize=True).reindex(discrete_values, fill_value=0)
            counts *= 100  # Convert to percentages

            # Bar plot for discrete attributes
            sns.barplot(x=counts.index, y=counts.values, palette="viridis", saturation=0.9)
            plt.title(f"Distribution of {attribute} (Discrete, per Case)", fontsize=16, fontweight="bold")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Percentage (%)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            for i, v in enumerate(counts.values):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10, fontweight="bold")
        
        # Handle continuous attributes
        elif any(rule['distribution']['type'] == 'normal' for rule in rules):
            # Calculate number of bins (using Sturges' rule)
            bins = int(np.ceil(np.log2(len(attribute_values))) + 1)

            # Histogram for continuous attributes
            sns.histplot(attribute_values, bins=bins, kde=True, color='mediumvioletred', alpha=0.7)
            plt.title(f"Distribution of {attribute} (Continuous, per Case)", fontsize=16, fontweight="bold")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Percentage (%)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Display percentages on the histogram bars
            n, bin_edges = np.histogram(attribute_values, bins=bins)
            percentages = (n / n.sum()) * 100
            for i in range(len(n)):
                plt.text(bin_edges[i] + (bin_edges[i+1] - bin_edges[i]) / 2, percentages[i] + 0.5, 
                         f"{percentages[i]:.1f}%", ha='center', fontsize=10, fontweight="bold")

        else:
            print(f"Unsupported distribution type for attribute '{attribute}'. Skipping.")
            continue

        # Add final touches
        plt.tight_layout()
        image_path = os.path.join('img', folder_name, f"{attribute}.png")
        plt.savefig(image_path)
        plt.show()


def mine_bpm(file_name, folder_name):
    print("Mining BPM...")
    pd.set_option('display.max_columns', None)
    file_path = os.path.join("raw_data", file_name)
    data_df = pm4py.read_xes(file_path)
    output_dir = os.path.join("img", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # the log is filtered on the top 5 variants
    data_df = pm4py.filter_variants_top_k(data_df, 15)

    # a directly - follows graph (DFG) is discovered from the log
    dfg, start_activities, end_activities = pm4py.discover_dfg(data_df)

    # a process tree is discovered using the inductive miner
    process_tree = pm4py.discover_process_tree_inductive(data_df)
    # the process tree is converted to an accepting Petri net
    petri_net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
    process_tree = pm4py.discover_process_tree_inductive(data_df)
    # the accepting Petri net is converted to a BPMN diagram
    bpmn_diagram = pm4py.convert_to_bpmn(petri_net, initial_marking, final_marking )

    pm4py.save_vis_dfg(dfg, start_activities, end_activities, os.path.join(output_dir, "dfg.png"), format='png')
    pm4py.save_vis_bpmn(bpmn_diagram, os.path.join(output_dir, "bpmn.png"), format='png')
    print("--------------------------------------------------------------------------------------------------")
