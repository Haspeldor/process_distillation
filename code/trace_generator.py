import random
random.seed(0)
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm 

@dataclass
class Event:
    activity: str
    timestamp: datetime

@dataclass
class Case:
    case_id: int
    attributes: Dict[str, Any]
    events: List[Event] = field(default_factory=list)

@dataclass
class ProcessModel:
    start_activity: str = None
    activities: List[str] = field(default_factory=list)
    transitions: Dict[str, List[Tuple[str, Dict[Tuple[str, Any], float]]]] = field(default_factory=dict)
    case_attribute_distribution: Dict[str, List[Tuple[Any, float]]] = field(default_factory=dict)

    def add_attribute(self, attribute_name: str, value_probabilites: List[Tuple[Any, float]]):
        self.case_attribute_distribution[attribute_name] = value_probabilites

    def add_activity(self, activity: str, start_activity=False):
        if activity not in self.activities:
            self.activities.append(activity)
            self.transitions[activity] = []
        if start_activity:
            self.start_activity = activity

    # Add a transition between two activities with specific probabilities based on case attribute values
    def add_transition(self, from_activity: str, to_activity: str, conditions: Dict[Tuple[str, Any], float]):
        if from_activity in self.activities and to_activity in self.activities:
            self.transitions[from_activity].append((to_activity, conditions))

    # Get the next activity based on case attributes and transition probabilities
    def get_next_activity(self, current_activity: str, case_attributes: Dict[str, Any]) -> Optional[str]:
        if current_activity not in self.transitions or not self.transitions[current_activity]:
            return None  # No further activities
        
        possible_transitions = self.transitions[current_activity]
        total_probability = 0
        weighted_choices = []

        # For each possible transition, evaluate the probability based on case attributes
        for next_activity, conditions in possible_transitions:
            # Default probability is 1
            probability = 1.0
            
            # Go through each condition and check if case attribute matches
            for (attr, value), prob in conditions.items():
                if case_attributes.get(attr) == value:
                    probability *= prob
            
            if probability > 0:
                weighted_choices.append((next_activity, probability))
                total_probability += probability

        # Normalize probabilities and randomly select next activity
        r = random.uniform(0, total_probability)
        cumulative_probability = 0
        for next_activity, probability in weighted_choices:
            cumulative_probability += probability
            if r <= cumulative_probability:
                return next_activity

        return None

@dataclass
class TraceGenerator:
    process_model: ProcessModel
    current_case_id: int = 0

    # generate event traces 
    def generate_traces(self, start_time: datetime = datetime.now(), num_cases: int = 1000, max_steps: int = 10) -> List[Event]:
        cases = []
        for _ in tqdm(range(num_cases), desc="generating data"):
            case = self.generate_case()
            current_activity = self.process_model.start_activity
            current_time = start_time

            for _ in range(max_steps):
                if not current_activity:
                    break

                # Generate event
                event = Event(activity=current_activity, timestamp=current_time)
                case.events.append(event)

                # Get next activity
                current_activity = self.process_model.get_next_activity(current_activity, case.attributes)

                # Increment time (arbitrary 1-10 minute increments)
                current_time += timedelta(minutes=random.randint(1, 10))
            
            cases.append(case)
            start_time += timedelta(minutes=random.randint(1, 10))

        return cases
    
    #Generate case with attributes based on provided distributions
    def generate_case(self, noise_attribute: float = 0.01) -> Case:
        case_id = self.current_case_id
        self.current_case_id += 1

        generated_attributes = {}

        for attr, value_probs in self.process_model.case_attribute_distribution.items():
            values, probabilities = zip(*value_probs)
            chosen_value = random.choices(values, probabilities)[0]
            if random.random() < noise_attribute:
                random_attr_value = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                generated_attributes[attr] = random_attr_value
            else:
                generated_attributes[attr] = chosen_value
        

        return Case(case_id = case_id, attributes=generated_attributes)
    
    # return list of the activities of the process_model
    def get_events(self) -> List[str]:
        return sorted(self.process_model.activities)

    # return a dict with the attribute names as keys and lists of possible attribute values as values
    def get_case_attribute_pools(self) -> Dict[str, List[any]]:
        return {key: sorted([item[0] for item in value]) for key, value in self.process_model.case_attribute_distribution.items()}

def build_process_model(model_name) -> ProcessModel:
    print("building process model:")
    process_model = ProcessModel()
    if model_name == "model_1":
        process_model.add_attribute("gender", [("male", 0.45),("female", 0.45), ("diverse", 0.1),])
        process_model.add_activity("start",start_activity=True)
        process_model.add_activity("casual background check")
        process_model.add_activity("exhaustive background check")
        process_model.add_activity("end")
        process_model.add_transition("start", "casual background check", conditions={
            ("gender", "male"): 0.8,
            ("gender", "female"): 0.3,  
            ("gender", "diverse"): 0.0 
        })
        process_model.add_transition("start", "exhaustive background check", conditions={
            ("gender", "male"): 0.2,
            ("gender", "female"): 0.7,  
            ("gender", "diverse"): 1.0 
        })
        process_model.add_transition("casual background check", "end", conditions={})
        process_model.add_transition("exhaustive background check", "end", conditions={})

    elif model_name == "model_2":
        process_model.add_attribute("gender", [("male", 0.45),("female", 0.45), ("diverse", 0.1),])
        process_model.add_attribute("age", [("young", 0.2),("middle aged", 0.5), ("old", 0.3),])
        process_model.add_activity("start",start_activity=True)
        process_model.add_activity("talk to applicant")
        process_model.add_activity("casual background check")
        process_model.add_activity("exhaustive background check")
        process_model.add_activity("accept request")
        process_model.add_activity("reject request")
        process_model.add_activity("end")
        process_model.add_transition("start", "talk to applicant", conditions={})
        process_model.add_transition("talk to applicant", "casual background check", conditions={
            ("gender", "male"): 0.8,
            ("gender", "female"): 0.3,  
            ("gender", "diverse"): 0.0 
        })
        process_model.add_transition("talk to applicant", "exhaustive background check", conditions={
            ("gender", "male"): 0.2,
            ("gender", "female"): 0.7,  
            ("gender", "diverse"): 1.0 
        })
        process_model.add_transition("casual background check", "accept request", conditions={
            ("gender", "male"): 0.7,
        })
        process_model.add_transition("casual background check", "reject request", conditions={
            ("gender", "male"): 0.3,
        })
        process_model.add_transition("exhaustive background check", "accept request", conditions={
            ("age", "young"): 0.3,
            ("age", "middle aged"): 0.6,  
            ("age", "old"): 0.8 
        })
        process_model.add_transition("exhaustive background check", "reject request", conditions={
            ("age", "young"): 0.7,
            ("age", "middle aged"): 0.4,  
            ("age", "old"): 0.2 
        })
        process_model.add_transition("accept request", "end", conditions={})
        process_model.add_transition("reject request", "end", conditions={})

    elif model_name == "model_3":
        process_model.add_attribute("gender", [("male", 0.45),("female", 0.45), ("diverse", 0.1),])
        process_model.add_activity("start",start_activity=True)
        process_model.add_activity("talk to applicant")
        process_model.add_activity("casual background check")
        process_model.add_activity("exhaustive background check")
        process_model.add_activity("accept request")
        process_model.add_activity("reject request")
        process_model.add_activity("end")
        process_model.add_transition("start", "talk to applicant", conditions={})
        process_model.add_transition("talk to applicant", "casual background check", conditions={
            ("gender", "male"): 0.8,
            ("gender", "female"): 0.3,  
            ("gender", "diverse"): 0.0 
        })
        process_model.add_transition("talk to applicant", "exhaustive background check", conditions={
            ("gender", "male"): 0.2,
            ("gender", "female"): 0.7,  
            ("gender", "diverse"): 1.0 
        })
        process_model.add_transition("casual background check", "accept request", conditions={
            ("gender", "male"): 0.8,
            ("gender", "female"): 0.6,  
            ("gender", "diverse"): 0.3 
        })
        process_model.add_transition("casual background check", "reject request", conditions={
            ("gender", "male"): 0.2,
            ("gender", "female"): 0.4,  
            ("gender", "diverse"): 0.7 
        })
        process_model.add_transition("exhaustive background check", "accept request", conditions={
            ("gender", "male"): 0.7,
            ("gender", "female"): 0.5,  
            ("gender", "diverse"): 0.3 
        })
        process_model.add_transition("exhaustive background check", "reject request", conditions={
            ("gender", "male"): 0.3,
            ("gender", "female"): 0.5,  
            ("gender", "diverse"): 0.7 
        })
        process_model.add_transition("accept request", "end", conditions={})
        process_model.add_transition("reject request", "end", conditions={})
    print(process_model)
    print("--------------------------------------------------------------------------------------------------")
    return process_model
