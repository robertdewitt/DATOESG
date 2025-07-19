import os
import logging 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# this is a class for storing RL agents so that they can be looped through for training, testing, printing, etc.

class ModelParking:
    def __init__(self, model_dir=None, cached=False,  models=None, env=None):
        """
        Initialize the ModelParking with an optional list of models.
        @param model_dir: Directory to save/load models. Defaults to "models".
        @param models: List of models to park.
        """
        self.model_dir = model_dir if model_dir is not None else "models"
        self.models = models if models is not None else []

        if cached:
            logging.info("ModelParking initialized with cached models.")
            # load cached models from the model directory
            for model_file in os.listdir(self.model_dir):
                if model_file.endswith('.zip'):
                    model_name = model_file[:-4]
                    model_path = os.path.join(self.model_dir, model_file)
                    # load zip file as model
                    try:
                        model = PPO.load(model_path, env=env, device="auto", print_system_info=True)
                        logging.debug(f"Model '{model_name}' loaded from cache.")
                    except Exception as e:
                        logging.error(f"Failed to load model '{model_name}': {e}")

        else:
            logging.info("ModelParking initialized without cached models.")

        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        

    def park_model(self, model, model_name=None, save=True, learn=False, steps=1000):
        """
        Park the model for storage, also save the model to disc
        @param model: The model to add.
        @param model_name: Optional name for the model. If None, uses the class name.
        """
        if model_name is None:
            model_name = model.__class__.__name__
        
        self.models.append((model_name, model))
        logging.debug(f"Model '{model_name}' added to parking lot.")

        if learn:
            logging.debug(f"Model '{model_name}' is learning for {steps} steps.")
            model.learn(steps)
        
        if save:
            model.save(f"{self.model_dir}/{model_name}")
            logging.debug(f"Model '{model_name}' saved to {self.model_dir}.")
        
    def get_model(self, model_name):
        """
        Retrieve a parked model by its name.
        @param model_name: Name of the model to retrieve.
        @return: The model if found, None otherwise.
        """
        for name, model in self.models:
            if name == model_name:
                logging.debug(f"Model '{model_name}' retrieved from parking lot.")
                return model
        
        logging.warning(f"Model '{model_name}' not found in parking lot.")
        return None
    
    def list_models(self):
        """
        List all parked models.
        @return: List of model names.
        """
        model_names = [name for name, _ in self.models]
        logging.debug(f"Parked models: {model_names}")
        return model_names
    
    def model_learn(self, model_name, steps=1000):
        """
        Call the learn method of a parked model.
        @param model_name: Name of the model to learn from.
        @param steps: Number of steps for learning.
        @return: Result of the learn method if successful, None otherwise.
        """
        model = self.get_model(model_name)
        if model is not None:
            logging.debug(f"Model '{model_name}' is learning for {steps} steps.")
            return model.learn(steps)
        
        logging.error(f"Cannot learn. Model '{model_name}' not found.")
        return None

    