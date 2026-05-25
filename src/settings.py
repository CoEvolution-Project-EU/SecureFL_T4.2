from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from yaml import safe_load

from avisence_datasets.poss.parser import Parser
from modules.loss.boundary_loss import BoundaryLoss
from modules.loss.Lovasz_Softmax import Lovasz_softmax


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
            "Loss-based Clustering",
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


class UseCase(BaseModel):
    name: str
    data_split: Literal["non-iid", "iid"] = "iid"
    data_config_path: str
    model_architecture_config_path: str
    data_dir: str

    data_config: Optional[dict] = None
    model_architecture_config: Optional[dict] = None
    parser: Optional["Parser"] = None

    criterion: Optional[nn.NLLLoss] = None
    lovasz: Optional[nn.Module] = None
    boundary_loss: Optional[nn.Module] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def init_configs(self):
        """Load configs after UseCase is constructed"""
        if self.name == "AVISENCE":
            # Load YAML configs
            with open(self.model_architecture_config_path) as f:
                self.model_architecture_config = safe_load(f)
            with open(self.data_config_path) as f:
                self.data_config = safe_load(f)
        else:
            raise ValueError(f"Unknown use case: {self.name}")
        return self

    def init_parser(self, local_batch_size: int):
        match self.name:
            case "AVISENCE":
                self.parser = Parser(
                    root=self.data_dir,
                    train_sequences=self.data_config["split"]["train"],
                    valid_sequences=self.data_config["split"]["valid"],
                    test_sequences=self.data_config["split"]["test"],
                    labels=self.data_config["labels"],
                    color_map=self.data_config["color_map"],
                    learning_map=self.data_config["learning_map"],
                    learning_map_inv=self.data_config["learning_map_inv"],
                    sensor=self.model_architecture_config["dataset"]["sensor"],
                    max_points=self.model_architecture_config["dataset"]["max_points"],
                    batch_size=local_batch_size,
                    workers=0,
                    gt=True,
                    shuffle_train=True,
                )
            case _:
                raise ValueError(f"Unknown use case: {self.name}")

    def init_loss_functions(self, device):
        match self.name:
            case "AVISENCE":
                """Setup loss functions"""
                # Calculate class weights from dataset
                epsilon_w = self.model_architecture_config["train"]["epsilon_w"]
                content = torch.zeros(len(self.data_config["learning_map_inv"]), dtype=torch.float)

                # Map content from original classes to learning classes
                for cl, freq in self.data_config["content"].items():
                    x_cl = int(cl)
                    # Map original class to learning class
                    if x_cl in self.data_config["learning_map"]:
                        mapped_cl = self.data_config["learning_map"][x_cl]
                        content[mapped_cl] += freq

                loss_w = 1 / (content + epsilon_w)
                loss_w[0] = 0  # Ignore unlabeled class

                # Loss functions - convert weights to double to match the log output type
                self.criterion = nn.NLLLoss(weight=loss_w.double()).to(device)
                self.lovasz = Lovasz_softmax(ignore=0).to(device)
                self.boundary_loss = BoundaryLoss().to(device)
            case _:
                raise ValueError(f"Unknown use case: {self.name}")


class Config(BaseModel):
    server: Server
    client: Client
    model: Model
    attack: Attack
    defence: Defence
    general: General
    backend: Backend
    use_case: Optional[UseCase] = None
    config_path: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @model_validator(mode="after")
    def init_use_case_parser(self):
        if self.use_case is not None:
            self.use_case.init_parser(local_batch_size=self.client.batch_size)
        return self

    @model_validator(mode="after")
    def init_use_case_loss_functions(self):
        if self.use_case is not None:
            # We don't have general.device in settings.py yet, we can default to cuda if available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.use_case.init_loss_functions(device=device)
        return self

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
config_file = FOLDER_DIR / "config.yaml"
settings = Config(config_file)
