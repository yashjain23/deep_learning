import pickle
import json

def load_parameters(save_path="parameters.pkl"):
    with open(save_path, "rb") as file:
        parameters = pickle.load(file)
    print(f"Parameters loaded from {save_path}")
    return parameters

parameters = load_parameters()


def export_parameters(parameters, file_name="model_parameters.json"):
    parameters_json = {key: value.tolist() for key, value in parameters.items()}
    with open(file_name, "w") as f:
        json.dump(parameters_json, f)
    print(f"Parameters exported to {file_name}")

# Example call
export_parameters(parameters)