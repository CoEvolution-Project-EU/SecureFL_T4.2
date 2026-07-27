import torch
from flwr.client import NumPyClient

from modules.utils import iouEval
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader, Subset

from src.models import ModelConfig, get_weights, set_weights
from src.settings import settings
from src.task import test, train


class FlowerClient(NumPyClient):
    """
    A custom Flower client for federated learning.

    This client maintains a local model instance and provides the standard fit and evaluate 
    interfaces for federated interaction. It simulates local data loading, applies optional 
    sensor-specific masking, and executes local attacks if marked as malicious.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        client_type: str,
        partition_id: int,
        train_loader=None,
        val_loader=None,
        client_indices=None,
        sensor_config=None,
        sequence_meta=None,
    ):
        self.model_config = model_config
        self.client_type = client_type
        self.partition_id = partition_id
        self.sensor_config = sensor_config
        self.sequence_meta = sequence_meta

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if settings.use_case is None or settings.use_case.name != "AVISENCE":
            raise ValueError("Only the AVISENCE use case is supported in this configuration.")


        self.client_indices = client_indices
        client_dataset = Subset(settings.use_case.parser.train_dataset, self.client_indices)
        self.train_loader = DataLoader(
            client_dataset, batch_size=settings.client.batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = settings.use_case.parser.get_valid_set()
        self.evaluator = iouEval(model_config.num_classes, device, ignore=0)

        self.model = model_config.model
        self.model.to(device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Trains the global model on the client's local data partition.

        Receives the global model parameters from the server, updates the local model, 
        and performs local epochs of training. If the client is malicious and the attack 
        is activated, adversarial manipulations are applied during this phase.

        :param parameters: The current global model parameters provided by the server.
        :param config: A configuration dictionary containing training instructions (e.g., learning rate, attack activation).
        :return: A tuple containing the updated local parameters, the number of training examples used, and a metrics dictionary.
        """
        attack_activated = bool(config["attack_activated"])
        lr = float(config["lr"])
        # Apply weights from global models (the whole model is replaced)
        set_weights(self.model, parameters)

        train(
            self.model,
            self.train_loader,
            client_type=self.client_type,
            lr=lr,
            model_config=self.model_config,
            attack_activated=attack_activated,
            partition_id=self.partition_id,
            sensor_config=self.sensor_config,
            sequence_meta=self.sequence_meta,
        )

        # Return locally-trained model and metrics
        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"id": self.partition_id, "client_type": self.client_type},
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        """
        Evaluates the global model on the client's local validation set.

        Overwrites the local model with the provided global parameters and computes 
        the loss, accuracy, and Jaccard index to assess the model's performance on 
        the client's localized data distribution.

        :param parameters: The current global model parameters provided by the server.
        :param config: A configuration dictionary detailing evaluation instructions.
        :return: A tuple containing the computed loss, the number of validation examples used, and a metrics dictionary.
        """
        set_weights(self.model, parameters)

        loss, accuracy, jaccard = test(
            self.model,
            self.val_loader,
            evaluator=self.evaluator,
            call_desc=f"Client {self.partition_id} Evaluation",
        )
        return (
            loss,
            len(self.val_loader.dataset),
            {"accuracy": accuracy},
        )
