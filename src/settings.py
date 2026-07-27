import os
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from yaml import safe_load

from avisence_datasets.poss.parser import Parser
from modules.loss.boundary_loss import BoundaryLoss
from modules.loss.Lovasz_Softmax import Lovasz_softmax


class Server(BaseModel):
    """
    Configuration for the federated learning server.
    
    Defines the aggregation strategy, fraction of clients to select, and 
    hyperparameters controlling the global training rounds.
    """
    strategy: str
    fraction_fit: float
    fraction_eval: float
    dataset_size: float
    num_rounds: int
    batch_size: int

    @field_validator("fraction_fit", "fraction_eval", "dataset_size")
    def validate_percentages(cls, value, info):
        """
        Validates that percentage fields are strictly between 0.0 and 1.0.
        
        :param value: The percentage value being validated.
        :param info: Validation context providing the field name.
        :return: The validated percentage value.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under server configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("batch_size", "num_rounds")
    def validate_positive(cls, value, info):
        """
        Validates that integer configuration fields are strictly positive.
        
        :param value: The integer value being validated.
        :param info: Validation context providing the field name.
        :return: The validated positive integer.
        """
        if value <= 0:
            raise ValueError(f"Under server configuration: {info.field_name} must be positive. Got {value}")
        return value

    @field_validator("strategy")
    def validate_server_strategy(cls, value: str, info: ValidationInfo):
        """
        Validates the server aggregation strategy against supported algorithms.
        
        :param value: The strategy name to validate (e.g., 'Mean', 'Krum').
        :param info: Validation context.
        :return: The validated strategy string.
        """
        strategy_types = [
            "Mean",
            "Krum",
            "Multi-Krum",
            "Bulyan",
            "Trimmed-Mean",
            "Median",
            "FedCluster",
            "FedGreed",
            "FedTruncate",
            "FL-Defender",
            "FoolsGold",
            "RFA",
        ]

        if value not in strategy_types:
            raise ValueError(f"Under server configuration: {info.field_name} must be in {strategy_types}. Got {value}")
        return value


class Client(BaseModel):
    """
    Configuration for federated learning clients.
    
    Specifies the total pool of clients and their local training hyperparameters.
    """
    num_clients: int
    batch_size: int
    local_epochs: int

    @field_validator("num_clients", "batch_size", "local_epochs")
    def validate_positive(cls, value, info):
        """
        Validates that client configuration fields are strictly positive.
        
        :param value: The integer value being validated.
        :param info: Validation context.
        :return: The validated positive integer.
        """
        if value <= 0:
            raise ValueError(f"Under client configuration: {info.field_name} must be positive. Got {value}")
        return value


class Dataset(BaseModel):
    """
    Configuration for the simulated dataset.
    """
    name: str

    @field_validator("name")
    def validate_dataset_name(cls, value: str, info: ValidationInfo):
        if value != "AVISENCE":
            raise ValueError(f"Under dataset configuration: {info.field_name} must be 'AVISENCE'." f" Got {value}")
        return value


class Model(BaseModel):
    """
    Configuration specifying the neural network architecture.
    """
    name: str

    @field_validator("name")
    def validate_model_name(cls, value: str, info: ValidationInfo):
        if value not in ("ResNet",):
            raise ValueError(
                f"Under model configuration: {info.field_name} must be 'ResNet' (for AVISENCE)." f" Got {value}"
            )
        return value


class Optimizer(BaseModel):
    """
    Configuration for the local training optimizer (e.g., SGD, Adam).
    """
    name: str = "sgd"
    lr: float = 0.1
    lr_decay_epochs: list[int] = 75
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = -1.0
    nesterov: bool = False

    @field_validator("lr")
    def validate_positive_lr(cls, value, info):
        if value <= 0:
            raise ValueError(f"Under optimizer configuration: {info.field_name} must be positive. Got {value}")
        return value


class Attack(BaseModel):
    """
    Configuration detailing malicious client attacks.
    
    Controls the type of attack, the number of colluding malicious clients, 
    and attack-specific scaling hyperparameters.
    """
    activation_round: int = 0
    num_malicious_clients: int = 0

    type: str | None = None

    # For gaussian noise attack
    mean: float = 0.0
    std: float = 1.0

    alie_z_max: float | None = 3.0
    epsilon: float = 0.2
    target_rank: int = 0
    dev_type: str = "std"

    @field_validator("dev_type")
    def validate_dev_type(cls, value: str, info: ValidationInfo):
        """
        Validates the vector type used for perturbation attacks.
        
        :param value: The perturbation type (e.g., 'std', 'sign').
        :param info: Validation context.
        :return: The validated string.
        """
        dev_types = ["std", "sign", "unit_vec"]
        if value not in dev_types:
            raise ValueError(f"Under attack configuration: {info.field_name} must be in {dev_types}. Got {value}")
        return value

    @field_validator("activation_round", "num_malicious_clients")
    def validate_positive(cls, value, info):
        """
        Validates that the attack activation round and client counts are non-negative.
        
        :param value: The integer or float value being validated.
        :param info: Validation context.
        :return: The validated numerical value.
        """
        if value < 0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be non-negative. Got {value}")
        return value

    @field_validator("type")
    def validate_attack_type(cls, value: str | None, info: ValidationInfo):
        """
        Validates the configured attack algorithm against supported types.
        
        :param value: The chosen attack (e.g., 'ALIE', 'IPM').
        :param info: Validation context.
        :return: The validated attack string.
        """
        attack_types = ["Gaussian", "IPM", "ALIE", "Semantic-Label-Flip", "Sign-Flip", "None"]
        if value not in attack_types + [None]:
            raise ValueError(f"Under attack configuration: {info.field_name} must be in {attack_types}. Got {value}")
        return value

    @model_validator(mode="after")
    def zero_malicious_clients_if_no_attack(self):
        """
        Automatically overrides num_malicious_clients to 0 when no attack is taking place.
        """
        if self.type == "None" or self.type is None:
            self.num_malicious_clients = 0
        return self


class Defence(BaseModel):
    """
    Configuration detailing server-side robust aggregation defenses.
    
    Includes hyperparameters for tuning defenses like Trimmed Mean, RFA, 
    FoolsGold, and FL-Defender.
    """
    activation_round: int = 0
    lbc_num_selected_clients: int = 0
    defence_dataset_percentage: float = 0.1
    beta: float = 0.2  # For Trimmed Mean Strategy only
    rfa_t: int = 5  # For RFA Strategy
    rfa_nu: float = 1e-6  # For RFA Strategy


    # For FoolsGold Strategy
    fg_use_memory: bool = True
    fg_memory_size: int = 10
    fg_epsilon: float = 1e-5

    # For FedTruncate Strategy
    num_selected_clients: int = 0
    B: float = 1.0
    B0: float = 0.0
    gamma: float = 1.0
    eps: float = 0.0

    @field_validator("beta")
    def validate_percentages(cls, value: float, info: ValidationInfo):
        """
        Validates that percentage fields associated with defense configurations are strictly between 0.0 and 1.0.
        
        :param value: The percentage value being validated.
        :param info: Validation context providing the field name.
        :return: The validated percentage value.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("activation_round", "lbc_num_selected_clients", "num_selected_clients")
    def must_be_non_negative(cls, value, info: ValidationInfo):
        if value < 0:
            raise ValueError(f"Under defence configuration: {info.field_name} must be non-negative. Got {value}")
        return value

    @field_validator("defence_dataset_percentage")
    def must_be_valid_percentage(cls, value, info: ValidationInfo):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Under defence configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value


    @field_validator("B", "B0", "gamma", "eps")
    def validate_non_negative_floats(cls, value: float, info: ValidationInfo):
        """
        Validates that specific float hyperparameters (e.g., FedTruncate thresholds) are non-negative.
        
        :param value: The float value to validate.
        :param info: Validation context.
        :return: The validated float value.
        """
        if value < 0.0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be non-negative. Got {value}")
        return value


class General(BaseModel):
    use_wandb: bool
    random_seed: int


class Backend(BaseModel):
    client_resources: dict[str, float]


class LENetAuxLossConfig(BaseModel):
    use: bool
    lamda: list[float]

class LENetTrainConfig(BaseModel):
    loss_class_weight_smoothing: float
    aux_loss: LENetAuxLossConfig
    residual: bool
    n_input_scans: int

class LENetPostParamsConfig(BaseModel):
    knn: int
    search: int
    sigma: float
    cutoff: float

class LENetKNNConfig(BaseModel):
    use: bool
    params: LENetPostParamsConfig

class LENetCRFConfig(BaseModel):
    use: bool

class LENetPostConfig(BaseModel):
    CRF: LENetCRFConfig
    KNN: LENetKNNConfig

class LENetImgPropConfig(BaseModel):
    width: int
    height: int

class LENetSensorConfig(BaseModel):
    name: str
    type: str
    fov_up: float
    fov_down: float
    img_prop: LENetImgPropConfig
    img_means: list[float]
    img_stds: list[float]
    n_input_scans: int
    residual: bool
    transform: bool
    use_normal: bool

class LENetDatasetConfig(BaseModel):
    labels: str
    scans: str
    max_points: int
    sensor: LENetSensorConfig

class LENetConfig(BaseModel):
    train: LENetTrainConfig
    post: LENetPostConfig
    dataset: LENetDatasetConfig

class UseCase(BaseModel):
    name: str
    data_split: Literal["iid", "sensor"] = "iid"
    sensor_profiles: Optional[dict] = None
    sensor_mapping: Optional[dict] = None
    data_config_path: str
    lenet_poss_config: LENetConfig
    data_dir: str

    data_config: Optional[dict] = None
    model_architecture_config: Optional[dict] = None
    parser: Optional[object] = None

    criterion: Optional[nn.NLLLoss] = None
    lovasz: Optional[nn.Module] = None
    boundary_loss: Optional[nn.Module] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def init_configs(self):
        """
        Dynamically loads secondary configuration files after the UseCase instance is constructed.
        
        This bridges the Pydantic structured configuration with the legacy dictionary-based 
        model configurations expected by the AVISENCE architecture.
        """
        if self.name == "AVISENCE":
            # Map lenet_poss_config to dictionary to preserve compatibility with dict-based codebase
            self.model_architecture_config = self.lenet_poss_config.model_dump()
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
                loss_class_weight_smoothing = self.model_architecture_config["train"]["loss_class_weight_smoothing"]
                content = torch.zeros(len(self.data_config["learning_map_inv"]), dtype=torch.float)

                # Map content from original classes to learning classes
                for cl, freq in self.data_config["content"].items():
                    x_cl = int(cl)
                    # Map original class to learning class
                    if x_cl in self.data_config["learning_map"]:
                        mapped_cl = self.data_config["learning_map"][x_cl]
                        content[mapped_cl] += freq

                loss_w = 1 / (content + loss_class_weight_smoothing)
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
    dataset: Dataset
    optimizer: Optimizer
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
                if "optimizer" not in config:
                    config.update({"optimizer": Optimizer()})
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


PROJECT_NAME = "FL Defense Project"
FOLDER_DIR = Path(__file__).parent.parent
config_name = os.getenv("config_file_name", "config")
if config_name.endswith(".yaml"):
    config_name = config_name[:-5]
elif config_name.endswith(".yml"):
    config_name = config_name[:-4]
config_file = FOLDER_DIR / f"{config_name}.yaml"

try:
    settings = Config(config_file)
except ValidationError as e:
    print("\n[Configuration Error]")
    for error in e.errors():
        print(f"❌ {error.get('msg', 'Validation Error')}")
    print()
    sys.exit(1)


def _global_value_error_handler(exc_type, exc_value, traceback):
    if issubclass(exc_type, ValueError):
        print("\n[Error]")
        print(f"❌ {exc_value}\n")
    else:
        sys.__excepthook__(exc_type, exc_value, traceback)


sys.excepthook = _global_value_error_handler

