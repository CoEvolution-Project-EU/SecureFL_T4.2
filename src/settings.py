import os
from pathlib import Path

from pydantic import BaseModel, ValidationInfo, field_validator
from yaml import safe_load


class Server(BaseModel):
    strategy: str
    fraction_fit: float
    fraction_eval: float
    dataset_size: float
    num_rounds: int
    batch_size: int
    beta: float = 0.2  # For Trimmed Mean Strategy only

    @field_validator("fraction_fit", "fraction_eval", "dataset_size", "beta")
    def validate_percentages(cls, value, info):
        """
        Validate individual percentage fields are between 0.0 and 1.0.
        :param value: Field validator
        :param info: Instance of Server class
        :return: Validated fields are between 0.0 and 1.0 or raises exception.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under server configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("batch_size", "num_rounds")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Server class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under server configuration: {info.field_name} must be positive. Got {value}")
        return value

    @field_validator("strategy")
    def validate_server_strategy(cls, value: str, info: ValidationInfo):
        """
        Validate strategy type is either Custom, Median or Krum.
        :param value: Field validator
        :param info: Instance of Strategy type
        :return: Validated attack type or raises exception.
        """
        strategy_types = [
            "FedGreed",
            "Mean",
            "Median",
            "Trimmed-Mean",
            "Krum",
            "Multi-Krum",
            "Bulyan",
            "Loss-based Clustering"
        ]
        if value not in strategy_types:
            raise ValueError(f"Under server configuration: {info.field_name} must be in {strategy_types}. Got {value}")
        return value


class Client(BaseModel):
    num_clients: int
    batch_size: int
    local_epochs: int

    @field_validator("num_clients", "batch_size", "local_epochs")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Client class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under client configuration: {info.field_name} must be positive. Got {value}")
        return value


class Model(BaseModel):
    name: str
    learning_rate: float

    @field_validator("learning_rate")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Model class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under model configuration: {info.field_name} must be positive. Got {value}")
        return value


class Attack(BaseModel):
    activation_round: int = 0
    num_malicious_clients: int = 0
    type: str = None
    # For BackDoor attack
    poison_rate: float = 0.5
    # For gaussian noise attack
    mean: float = 0.0
    std: float = 1.0

    @field_validator("activation_round", "num_malicious_clients")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Attack class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be positive. Got {value}")
        return value

    @field_validator("type")
    def validate_attack_type(cls, value: str, info: ValidationInfo):
        """
        Validate attack type is either Label Flip, Byzantine Attack or no attack at all (None).
        :param value: Field validator
        :param info: Instance of Attack type
        :return: Validated attack type or raises exception.
        """
        attack_types = ["Label Flip", "Gaussian Noise", "Sign Flip"]
        if value not in attack_types + [None]:
            raise ValueError(f"Under attack configuration: {info.field_name} must be in {attack_types}. Got {value}")
        return value


class Defence(BaseModel):
    activation_round: int = 0
    num_selected_clients: int = 0
    server_dataset_percentage: float = 1.0

    @field_validator("server_dataset_percentage")
    def validate_percentages(cls, value: float, info: ValidationInfo):
        """
        Validate individual percentage fields are between 0.0 and 1.0.
        :param value: Field validator
        :param info: Instance of Defence class
        :return: Validated fields are between 0.0 and 1.0 or raises exception.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("activation_round", "num_selected_clients")
    def validate_positive(cls, value: int, info: ValidationInfo):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Defence class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be positive. Got {value}")
        return value


class General(BaseModel):
    use_wandb: bool
    random_seed: int


class Backend(BaseModel):
    client_resources: dict[str, float]


class Config(BaseModel):
    server: Server
    client: Client
    model: Model
    attack: Attack
    defence: Defence
    general: General
    backend: Backend
    config_path: Path

    def __init__(self, config_path: Path) -> None:
        if config_path.is_file():
            with open(config_path) as f:
                config = safe_load(f)
                if "attack" not in config:
                    config.update({"attack": Attack()})
                if "defence" not in config:
                    config.update({"defence": Defence()})
            config.update({"config_path": config_path})
            super().__init__(**config)
        else:
            raise FileNotFoundError("Error: yaml config file not found.")

    @field_validator("attack")
    def validate_malicious_users(cls, value: Attack, info: ValidationInfo):
        """
        Check that the number of malicious users is valid.
        It should not exceed the total number of clients.
        :param value: Instance of Attack class
        :param info: Instance of Config class
        :return: Validated number of malicious users or raise exception
        """
        if "client" not in info.data.keys():
            raise ValueError("Client arguments are not properly defined.")
        num_clients = info.data["client"].num_clients
        if num_clients < value.num_malicious_clients:
            raise ValueError(
                f"Number of malicious clients ({value.num_malicious_clients}) cannot exceed "
                f"total number of clients ({num_clients}). "
            )
        return value

    @field_validator("attack", "defence")
    def validate_activation_round(cls, value: Attack | Defence, info: ValidationInfo):
        """
        Check that the activation round (Attack or Defence attribute) is valid.
        It should not exceed the total number of FL rounds.
        :param value: Instance of Attack or Defence class
        :param info: Instance of Config class
        :return: Validated activation round or raise exception
        """
        if "server" not in info.data.keys():
            raise ValueError("Server arguments are not properly defined.")
        num_rounds = info.data["server"].num_rounds
        activation_round = value.activation_round
        if num_rounds < activation_round:
            raise ValueError(
                f"Activation round for '{info.field_name}' cannot exceed total number of FL rounds ({num_rounds}). "
                f"Got activation round={activation_round}"
            )
        return value


PROJECT_NAME = "FedGreed FL Defense Project"
FOLDER_DIR = Path(__file__).parent.parent
config_file = FOLDER_DIR / f"configs/{os.getenv('config_file_name', 'config')}.yaml"
settings = Config(config_file)
