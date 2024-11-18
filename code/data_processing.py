import os
import pickle
import numpy as np
import pandas as pd

from typing import List, Dict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from trace_generator import Event, Case, TraceGenerator
from tqdm import tqdm 
from main import print_samples

# Preprocessing class
class DataProcessor:
    def __init__(self, trace_generator, n_gram=2, padding=True):
        self.n_gram = n_gram
        self.padding = padding
        self.trace_generator = trace_generator
        self.events = self.trace_generator.get_events()
        self.output_event_encoder = self.create_event_one_hot_encoder()
        if self.padding:
            self.events += ["<PAD>"]
        self.event_encoder = self.create_event_one_hot_encoder()
        self.attribute_encoders = self.create_case_attribute_one_hot_encoders()
        self.class_names = self.create_class_names()
        self.feature_names = self.create_feature_names()
        self.feature_indices = self.create_feature_indices()

    def create_class_names(self):
        class_names = self.event_encoder.categories_[0]
        try:
            class_names = class_names[class_names != "<PAD>"] 
        except ValueError:
            pass 
        return class_names

    def create_feature_names(self):
        n = self.n_gram
        event_pool = self.event_encoder.categories_[0]
        attribute_pools = self.trace_generator.get_case_attribute_pools()
        feature_names = []

        for index in range(n, 0, -1):
            for event in event_pool:
                feature_names.append(f"-{index}. Event = {event}")
        for attribute_name, possible_values in attribute_pools.items():
            for value in possible_values:
                feature_names.append(f"Attribute {attribute_name} = {value}")
        return feature_names

    def create_feature_indices(self):
        n = self.n_gram
        num_events = len(self.event_encoder.categories_[0])
        attribute_pools = self.trace_generator.get_case_attribute_pools()
        feature_indices = {}
        index = num_events * n

        for attribute_name, possible_values in attribute_pools.items():
            num_values = len(possible_values)
            feature_indices[attribute_name] = list(range(index, index + num_values))
            index += num_values
        return feature_indices

    # add padding to the case to enable using the first events for learning
    def pad_case(self, case: Case):
            if len(case.events) > 0:
                first_timestamp = case.events[0].timestamp
                pad_event = Event(activity="<PAD>",timestamp=first_timestamp)
                case.events = [pad_event] * self.n_gram + case.events

    # create one-hot encoder for the events
    def create_event_one_hot_encoder(self) -> OneHotEncoder:
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoder.fit(np.array(self.events).reshape(-1, 1))
        return one_hot_encoder

    # create a one-hot encoder for each case attribute
    def create_case_attribute_one_hot_encoders(self) -> Dict[str, OneHotEncoder]:
        attribute_encoders = {}
        for attr_name, attr_values in self.trace_generator.get_case_attribute_pools().items():
            one_hot_encoder = OneHotEncoder(sparse_output=False)
            one_hot_encoder.fit(np.array(list(attr_values)).reshape(-1, 1))
            attribute_encoders[attr_name] = one_hot_encoder
        return attribute_encoders

    # encode case attributes
    def encode_case_attributes(self, case: Case):
        encoded_attributes = []
        for attr_name, attr_value in case.attributes.items():
            encoder = self.attribute_encoders[attr_name]
            encoded_attr = encoder.transform([[attr_value]])[0]
            encoded_attributes.extend(encoded_attr)
        return np.array(encoded_attributes)

    # encode n-gram of events
    def encode_event_n_gram(self, n_gram: List[Event]):
        assert(len(n_gram) == self.n_gram)
        return np.array([self.event_encoder.transform([[event.activity]])[0] for event in n_gram]).flatten()
                
    # generate input-output pairs for neural network
    def process_data(self, cases: List[Case]):
        X, y = [], []
        for case in tqdm(cases, desc="processing data"):
            if self.padding:
                self.pad_case(case)
            events = case.events
            encoded_attributes = self.encode_case_attributes(case)
            
            # generate n-grams and the next event
            for i in range(len(events) - self.n_gram):
                # combine the encoded n_gram with the case attributes
                n_gram = events[i:i+self.n_gram]
                n_gram_encoded = self.encode_event_n_gram(n_gram) 
                encoded_sample = np.concatenate([n_gram_encoded, encoded_attributes])
                X.append(encoded_sample)

                # encode the next event as the target (one-hot encoded)
                next_event = events[i+self.n_gram]
                encoded_target = self.output_event_encoder.transform([[next_event.activity]])[0]
                y.append(encoded_target)
        
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

def load_data(folder_name, file_name):
    #print(f"loading {file_name}...")
    file_path = os.path.join('processed_data', folder_name, file_name)
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def save_data(data, folder_name, file_name):
    #print(f"saving {file_name}...")
    full_path = os.path.join('processed_data', folder_name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def generate_processed_data(process_model, num_cases, n_gram, folder_name=None):
    print("generating event traces:")
    trace_generator = TraceGenerator(process_model=process_model)
    cases = trace_generator.generate_traces(num_cases=num_cases)
    print("example trace:")
    print(cases[:1])
    print("event pool:")
    print(trace_generator.get_events())
    print("attribute pools:")
    print(trace_generator.get_case_attribute_pools())
    print("--------------------------------------------------------------------------------------------------")

    print("processing nn data:")
    #data_processor = DataProcessor(trace_generator=trace_generator, n_gram=n_gram)
    df = cases_to_dataframe(cases)
    X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices = process_df(df, ["gender", "problems"], n_gram=n_gram, folder_name=folder_name)
    print("example nn input:")
    print(X_train[:1])
    print("example nn output:")
    print(y_train[:1])
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
        save_data(process_model.critical_decisions, folder_name, "critical_decisions.pkl")

    return X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices, process_model.critical_decisions

def create_feature_names(event_pool, attribute_pools, n_gram):
    feature_names = []
    # Add event features for each n-gram step
    for index in range(n_gram, 0, -1):
        for event in sorted(event_pool):  # Ensure consistent order
            feature_names.append(f"-{index}. Event = {event}")
    # Add attribute features
    for attribute_name, possible_values in sorted(attribute_pools.items()):
        for value in sorted(possible_values):  # Ensure consistent order
            feature_names.append(f"Attribute {attribute_name} = {value}")
    return feature_names

def create_feature_indices(event_pool, attribute_pools, n_gram):
    num_events = len(sorted(event_pool))  # Sorted event pool
    feature_indices = {}
    index = num_events * n_gram  # Offset after all event features
    for attribute_name, possible_values in sorted(attribute_pools.items()):
        num_values = len(sorted(possible_values))  # Sorted attribute pool
        feature_indices[attribute_name] = list(range(index, index + num_values))
        index += num_values
    return feature_indices
    
def create_attribute_pools(df, case_attributes):
    attribute_pools = {}
    for attr in case_attributes:
        if attr in df.columns:
            attribute_pools[attr] = sorted(df[attr].dropna().unique().tolist())  # Ensure sorted order
        else:
            raise KeyError(f"Attribute '{attr}' is not in the DataFrame.")
    return attribute_pools 

def cases_to_dataframe(cases: List[Case]) -> pd.DataFrame:
    """
    Converts a list of Case objects into a pandas DataFrame with columns:
    'case_id', 'activity', and one column for each case attribute
    """
    rows = []
    for case in cases:
        for event in case.events:
            row = {
                'case_id': case.case_id,
                'activity': event.activity,
            }
            row.update(case.attributes)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def csv_to_df(file_name):
    file_path = os.path.join('raw_data', file_name)
    return pd.read_csv(file_path)
    

def process_df(df, case_attributes, n_gram=3, folder_name=None, test_size=0.2):
    """Processes dataframe data for neural network training"""
    # keep only specified attributes
    standard_attributes = ['case_id', 'activity']
    attributes_to_include = standard_attributes + case_attributes
    print(attributes_to_include)
    df = df[attributes_to_include]

    # create the meta-information
    class_names = sorted(df["activity"].unique().tolist() + ["<PAD>"])
    attribute_pools = create_attribute_pools(df, case_attributes)
    feature_names = create_feature_names(class_names, attribute_pools, n_gram)
    feature_indices = create_feature_indices(class_names, attribute_pools, n_gram)

    # one-hot encode activities
    activity_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[class_names])
    activity_ohe = activity_encoder.fit_transform(df[['activity']])
    pad_activity_idx = activity_encoder.categories_[0].tolist().index("<PAD>")
    
    # one-hot encode case attributes dynamically
    attribute_encoders = {}
    attributes_ohe_dict = {}

    for attr in case_attributes:
        print(attr)
        print(attribute_pools[attr])
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[attribute_pools[attr]])
        attributes_ohe_dict[attr] = encoder.fit_transform(df[[attr]])
        attribute_encoders[attr] = encoder

    # group by case_id and create sequences
    grouped = df.groupby('case_id')
    cases = []

    for case_id, group in grouped:
        activities = activity_encoder.transform(group[['activity']])
        attributes = {attr: attribute_encoders[attr].transform(group[[attr]]) for attr in case_attributes}
        cases.append((activities, attributes))

    # Generate n-grams with padding
    X, y = [], []
    pad_activity = activity_encoder.transform([["<PAD>"]])  # Get one-hot for <PAD>
    pad_attributes = {attr: np.zeros((1, attributes_ohe_dict[attr].shape[1])) for attr in case_attributes}

    for activities, attributes in cases:
        # Pad activities
        padded_activities = np.vstack([pad_activity] * n_gram + [activities])
        
        # Pad attributes
        padded_attributes = {attr: np.vstack([pad_attributes[attr]] * n_gram + [attributes[attr]])
                            for attr in sorted(attributes)}

        for i in range(len(activities)):  # Start from 0 and include all real activities
            x_activities = padded_activities[i:i + n_gram]
            x_attributes = np.hstack([padded_attributes[attr][i + n_gram - 1] for attr in case_attributes])
            x_attributes = np.hstack([attributes[attr][i - n_gram + 1] for attr in case_attributes])
            x_combined = np.hstack([x_activities.flatten(), x_attributes])  # Combine activities and attributes
            
            y_next_activity = activities[i]  # Predict the actual next activity
            X.append(x_combined)
            y.append(y_next_activity)
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    #print_samples(X, y, class_names, feature_names)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

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

    return X_train, X_test, y_train, y_test, class_names, feature_names, feature_indices