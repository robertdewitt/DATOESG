import os
import logging 
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize
import json, importlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def _class_path(obj_or_cls):
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    return f"{cls.__module__}.{cls.__name__}"

def _import_class(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)

@dataclass
class ModelManifest:
    name: str
    model_type: str          # "sb3" | "torch_module" | custom
    class_path: str          # dotted path for torch modules
    extra: Dict[str, Any]    # arbitrary metadata (e.g., fitness, descriptors)
    file_stem: str           # base filename without extension
    version: int = 1


# this is a class for storing RL agents so that they can be looped through for training, testing, printing, callbacks to tensorboard,  etc.

class ModelParking:
    """
    Unified parking for SB3 models (.zip) AND Torch nn.Module models (.pt + manifest).
    - Backwards compatible with your old __init__(model_dir=None, cached=False, models=None)
    - Adds serializer registry for future types (e.g., Mamba actor-critic)
    """


    def __init__(self, model_dir=None, cached=False,  models=None, env=None, device="auto"):
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
                        model = PPO.load(model_path, env=env, device=device, print_system_info=True)
                        logging.debug(f"Model '{model_name}' loaded from cache.")
                        self.models.append((model_name, model))
                    except Exception as e:
                        logging.error(f"Failed to load model '{model_name}': {e}")

        else:
            logging.info("ModelParking initialized without cached models.")

        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        

    def park_model(self, model, model_name=None, save=True, learn=False, steps=1000, callbacks=None, normalized=False, env=None):
        """
        Park the model for storage, also save the model to disc
       
        Args:
            model: The model to add.
            model_name: Optional name for the model. If None, uses the class name.
            save: Whether to save the model to disk.
            learn: Whether to train the model.
            steps: Number of training steps.
            callbacks: List of callbacks to use during training.
            normalized: Whether to normalize the environment.
            env: The environment to use for normalization.
            model_dir: The directory to save the model to.

        Returns:
        """
        if model_name is None:
            model_name = model.__class__.__name__
        
        self.models.append((model_name, model))
        logging.debug(f"Model '{model_name}' added to parking lot.")

        if learn:
            logging.debug(f"Model '{model_name}' is learning for {steps} steps.")
            if callbacks:
                # Create callback list if multiple callbacks provided
                if isinstance(callbacks, list):
                    callback_list = CallbackList(callbacks)
                else:
                    callback_list = callbacks
                model.learn(steps, callback=callback_list)
            else:
                model.learn(steps)
        
        if save:
            model.save(f"{self.model_dir}/{model_name}")
            logging.debug(f"Model '{model_name}' saved to {self.model_dir}.")
            if normalized:
                vecnorm: VecNormalize = env
                vecnorm.save(f"{self.model_dir}/{model_name}_vecnorm.pkl")
        
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

    def load_vecnorm(self, model_name, env=None):
        """
        Load the VecNormalize object for a model.
        @param model_name: Name of the model to load the VecNormalize for.
        @return: The VecNormalize object if found, None otherwise.
        """
        vecnorm = VecNormalize.load(f"{self.model_dir}/{model_name}_vecnorm.pkl", env)
        return vecnorm  
    
    def list_models(self):
        """
        List all parked models.
        @return: List of model names.
        """
        model_names = [name for name, _ in self.models]
        logging.debug(f"Parked models: {model_names}")
        return model_names
    
    def model_learn(self, model_name, steps=1000, callbacks=None):
        """
        Call the learn method of a parked model.
        @param model_name: Name of the model to learn from.
        @param steps: Number of steps for learning.
        @param callbacks: List of callbacks to use during training.
        @return: Result of the learn method if successful, None otherwise.
        """
        model = self.get_model(model_name)
        if model is not None:
            logging.debug(f"Model '{model_name}' is learning for {steps} steps.")
            if callbacks:
                # Create callback list if multiple callbacks provided
                if isinstance(callbacks, list):
                    callback_list = CallbackList(callbacks)
                else:
                    callback_list = callbacks
                return model.learn(steps, callback=callback_list)
            else:
                return model.learn(steps)
        
        logging.error(f"Cannot learn. Model '{model_name}' not found.")
        return None

    