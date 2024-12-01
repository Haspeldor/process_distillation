import os
import pickle
import numpy as np
import pandas as pd
import pm4py

from typing import List, Dict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from trace_generator import Event, Case, TraceGenerator
from tqdm import tqdm 

from main import print_samples

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

def load_xes_to_df(file_name, folder_name=None):
    file_path = os.path.join("raw_data", file_name)
    df = pm4py.read_xes(file_path)
    df.rename(columns={'case:concept:name': 'case_id', 'concept:name': 'activity'}, inplace=True)
    columns = ['case_id', 'activity'] + [col for col in df.columns if col not in ['case_id', 'activity']]
    df = df[columns]
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