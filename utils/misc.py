import datetime
import os

from transformers import PreTrainedTokenizer

from models.helper import Config
from utils.lr_utils.cosine_annealing_lr import CosineAnnealing
from utils.lr_utils.exp_decay_lr import ExpDecay
from utils.lr_utils.lr_scheduler import LearningRateScheduler


def get_tokenizer(model_type: str) -> PreTrainedTokenizer:
    """
    Get the tokenizer based on the model_type.

    Args:
        model_type (str): Name of the model.

    Raises:
        NotImplementedError: If unsupported model_type is provided then this
            exception will be thrown.

    Returns:
        PreTrainedTokenizer: Returns the tokenizer instance for a given model.
    """
    if model_type == "gpt":
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise NotImplementedError(
            f"No tokenizer implemented for model type: {model_type}"
        )


def lr_scheduler_factory(
    scheduler_type: str, *args, **kwargs
) -> LearningRateScheduler:
    """Function to get learning rate scheduler based on provided type.

    Args:
        scheduler_type (str): Type of scheduler.

    Raises:
        NotImplementedError: If the type of scheduler is not implemented then this
            will be raised.

    Returns:
        LearningRateScheduler: LR Scheduler instance.
    """
    mapping = {"cosine": CosineAnnealing, "exp": ExpDecay}

    scheduler_class = mapping.get(scheduler_type, None)
    if scheduler_class is None:
        raise NotImplementedError(
            f"Scheduler of type: {scheduler_type} is not implemented."
        )

    return scheduler_class(*args, **kwargs)


def init_wandb(
    train_config: Config, model_config: Config, resume_wandb_id: int
) -> None:
    """Initiate the weights and bias tracking. To be called at the start of experiment.

    Args:
        train_config (Config): Config instance representing training parameters.
        model_config (Config): Config instance representing model architecture
            parameters.
        resume_wandb_id (int): Weights and Bias tracking id to be reused in
            case of resuming training. Defaults to None.
    """
    import wandb

    config_dict = {**train_config.to_dict(), **model_config.to_dict()}
    wandb.init(
        project="TinyLLM",
        config=config_dict,
        resume="allow",
        id=resume_wandb_id,
    )


def get_exp_path(base_dir: str) -> str:
    """Function to get the directory to same the experiment related data.

    Args:
        base_dir (str): Directory to store all experiments.

    Returns:
        str: Path for current experiment.
    """
    start_time = datetime.datetime.now()
    exp_name = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    cur_exp_path = os.path.join(base_dir, exp_name)
    os.makedirs(cur_exp_path, exist_ok=True)
    return cur_exp_path
