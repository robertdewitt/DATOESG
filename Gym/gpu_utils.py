
import torch
import logging



def get_torch_device(log_level=logging.INFO):
    """
    Returns the appropriate PyTorch device based on availability.
    """
    logging.basicConfig(level=log_level)

    if torch.backends.mps.is_available():
        logging.info("Using Apple Metal Performance Shaders (MPS)")
        return 'mps'
    elif torch.cuda.is_available():
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA Device: {torch.cuda.current_device()}")
        return 'cuda'
    else:
        logging.info("No GPU available, using CPU")
        return 'cpu'
    
def set_random_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
