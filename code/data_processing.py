import os
import pickle
import numpy as np
import pandas as pd

from typing import List, Dict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from trace_generator import Event, Case, TraceGenerator
from tqdm import tqdm 

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
    data_processor = DataProcessor(trace_generator=trace_generator, n_gram=n_gram)
    X_train, X_test, y_train, y_test = data_processor.process_data(cases)
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
        save_data(data_processor.class_names, folder_name, "class_names.pkl")
        save_data(data_processor.feature_names, folder_name, "feature_names.pkl")
        save_data(data_processor.feature_indices, folder_name, "feature_indices.pkl")
        save_data(process_model.critical_decisions, folder_name, "critical_decisions.pkl")

    return X_train, X_test, y_train, y_test, data_processor.class_names, data_processor.feature_names, data_processor.feature_indices, process_model.critical_decisions

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
    

def process_csv_for_nn(df, case_attributes, n_gram=3, folder_name=None, num_cases=1000, test_size=0.2):
    """Processes dataframe data for neural network training"""
    # Keep only specified attributes
    standard_attributes = ['case_id', 'activity']
    attributes_to_include = standard_attributes + case_attributes
    df = df[attributes_to_include]
    
    case_attribute_encoders = {}
    encoded_case_attributes = {}

    for attr in case_attributes:
        if attr in df.columns:
            encoder = OneHotEncoder(sparse=False)
            encoded_case_attributes[attr] = encoder.fit_transform(df[[attr]])
            case_attribute_encoders[attr] = encoder

    # Event encoding (OneHotEncoder for activity column)
    event_encoder = OneHotEncoder(sparse=False)
    event_encoded = event_encoder.fit_transform(df[['activity']])

    # Prepare data storage
    X, y = [], []

    # Generate n-grams for each case
    case_ids = df['case_id'].unique()
    for case_id in case_ids:
        case_data = df[df['case_id'] == case_id]
        
        # Get case-level attribute encoding
        encoded_case_attributes_combined = []
        for attr in case_attributes:
            if attr in encoded_case_attributes:
                encoded_case_attributes_combined.extend(encoded_case_attributes[attr][case_data.index[0]])

        # Get n-grams and output labels
        events = case_data['activity'].tolist()

        for i in range(len(events) - n_gram):
            # Encode n-gram of events
            n_gram_events = events[i:i + n_gram]
            n_gram_encoded = np.array([event_encoder.transform([[activity]])[0] for activity in n_gram_events]).flatten()
            
            # Combine n-gram encoding with case attributes
            encoded_sample = np.concatenate([n_gram_encoded, encoded_case_attributes_combined])
            X.append(encoded_sample)

            # Encode the next event as the target
            next_event = events[i + n_gram]
            encoded_target = event_encoder.transform([[next_event]])[0]
            y.append(encoded_target)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    return X_train, X_test, y_train, y_test