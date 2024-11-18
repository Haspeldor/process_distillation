import random
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
class Decision:
    attributes: List[str]
    possible_events: List[str]
    to_remove: bool = True


@dataclass
class ProcessModel:
    start_activity: str = None
    activities: List[str] = field(default_factory=list)
    transitions: Dict[str, List[Tuple[str, Dict[Tuple[str, Any], float]]]] = field(default_factory=dict)
    case_attribute_distribution: Dict[str, List[Tuple[Any, float]]] = field(default_factory=dict)
    critical_decisions: List[Decision] = field(default_factory=list)

    def add_attribute(self, attribute_name: str, value_probabilites: List[Tuple[Any, float]]):
        self.case_attribute_distribution[attribute_name] = value_probabilites
    
    def add_critical_decision(self, attributes, possible_events, to_remove):
        self.critical_decisions.append(Decision(attributes=attributes, possible_events=possible_events, to_remove=to_remove))

    def add_activity(self, activity: str, start_activity=False):
        if activity not in self.activities:
            self.activities.append(activity)
            self.transitions[activity] = []
        if start_activity:
            self.start_activity = activity

    # Add a transition between two activities with specific probabilities based on case attribute values
    def add_transition(self, from_activity: str, to_activity: str, conditions: Dict[Tuple[str, Any], float] = field(default_factory=dict)):
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
    noise_transition: float = 0.010
    noise_event: float = 0.01
    noise_time: float = 0.00 
    noise_attribute: float = 0.001

    # generate event traces 
    def generate_traces(self, start_time: datetime = datetime.now(), num_cases: int = 1000, max_steps: int = 100) -> List[Case]:
        random.seed(0)
        cases = []
        for _ in tqdm(range(num_cases), desc="generating data"):
            case = self.generate_case()
            current_activity = self.process_model.start_activity
            current_time = start_time

            for _ in range(max_steps):
                if not current_activity:
                    break

                # Determine event name with noise_event probability
                if random.random() < self.noise_event:
                    event_name = ''.join(random.choices(string.ascii_letters, k=10))
                else:
                    event_name = current_activity

                # Create event with either original or randomized activity name
                event = Event(activity=event_name, timestamp=current_time)
                case.events.append(event)

                # Choose next activity with noise_transition probability
                if random.random() < self.noise_transition:
                    current_activity = random.choice(self.process_model.activities)  # Randomly pick an activity
                else:
                    current_activity = self.process_model.get_next_activity(current_activity, case.attributes)

                # Increment time, with noise_time probability adding random offset
                if random.random() < self.noise_time:
                    # Random offset of up to ±2 days and ±30 minutes
                    time_offset = timedelta(days=random.randint(-2, 2), minutes=random.randint(-30, 30))
                    current_time += time_offset
                else:
                    current_time += timedelta(minutes=random.randint(1, 10))  # Normal increment

            cases.append(case)
            start_time += timedelta(minutes=random.randint(1, 10))  # Increment start time for the next case

        return cases
    
    #Generate case with attributes based on provided distributions
    def generate_case(self) -> Case:
        case_id = self.current_case_id
        self.current_case_id += 1

        generated_attributes = {}

        for attr, value_probs in self.process_model.case_attribute_distribution.items():
            values, probabilities = zip(*value_probs)
            chosen_value = random.choices(values, probabilities)[0]
            if random.random() < self.noise_attribute:
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

    elif model_name == "cc":
        process_model.add_attribute("gender", [("male", 0.5),("female", 0.5)])
        process_model.add_attribute("problems", [("true", 0.5),("false", 0.5)])
        process_model.add_activity("start",start_activity=True)
        process_model.add_activity("register")
        process_model.add_activity("asses eligibility")
        process_model.add_activity("collect history")
        process_model.add_activity("prostate screening")
        process_model.add_activity("mammary screening")
        process_model.add_activity("explain diagnosis")
        process_model.add_activity("discuss options")
        process_model.add_activity("inform prevention")
        process_model.add_activity("bill patient")
        process_model.add_activity("refuse screening")
        process_model.add_activity("end")
        process_model.add_transition("start", "register", conditions={})
        process_model.add_transition("register", "asses eligibility", conditions={})
        process_model.add_transition("asses eligibility", "collect history", conditions={
            ("gender", "male"): 0.7,
            ("gender", "female"): 0.3,  
        })
        process_model.add_transition("asses eligibility", "refuse screening", conditions={
            ("gender", "male"): 0.3,
            ("gender", "female"): 0.7,  
        })
        process_model.add_transition("collect history", "prostate screening", conditions={
            ("gender", "male"): 1,
            ("gender", "female"): 0,  
        })
        process_model.add_transition("collect history", "mammary screening", conditions={
            ("gender", "male"): 0,
            ("gender", "female"): 1,  
        })
        process_model.add_transition("prostate screening", "explain diagnosis", conditions={
            ("problems", "true"): 1,
            ("problems", "false"): 0,  
        })
        process_model.add_transition("prostate screening", "inform prevention", conditions={
            ("problems", "true"): 0,
            ("problems", "false"): 1,  
        })
        process_model.add_transition("mammary screening", "explain diagnosis", conditions={
            ("problems", "true"): 1,
            ("problems", "false"): 0,  
        })
        process_model.add_transition("mammary screening", "inform prevention", conditions={
            ("problems", "true"): 0,
            ("problems", "false"): 1,  
        })
        process_model.add_transition("explain diagnosis", "discuss options", conditions={})
        process_model.add_transition("discuss options", "bill patient", conditions={})
        process_model.add_transition("inform prevention", "bill patient", conditions={})
        process_model.add_transition("bill patient", "end", conditions={})
        process_model.add_transition("refuse screening", "end", conditions={})
        process_model.add_critical_decision(attributes=["gender"], possible_events=["prostate screening", "mammary screening"], to_remove=False)
        process_model.add_critical_decision(attributes=["gender"], possible_events=["collect history", "refuse screening"], to_remove=True)

    print(process_model)
    print("--------------------------------------------------------------------------------------------------")
    return process_model
