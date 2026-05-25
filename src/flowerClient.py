import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader, Subset

from modules.utils import iouEval
from src.models import ModelConfig, get_weights, set_weights
from src.settings import settings
from src.task import test, train


class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        client_type: str,
        partition_id: int,
        train_loader=None,
        val_loader=None,
        client_indices=None,
    ):
        self.model_config = model_config
        self.client_type = client_type
        self.partition_id = partition_id

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if settings.use_case is not None and settings.use_case.name == "AVISENCE":
            self.client_indices = client_indices
            client_dataset = Subset(settings.use_case.parser.train_dataset, self.client_indices)
            self.train_loader = DataLoader(
                client_dataset, batch_size=settings.client.batch_size, shuffle=True, drop_last=True
            )
            self.val_loader = settings.use_case.parser.get_valid_set()
            self.evaluator = iouEval(model_config.num_classes, device, ignore=0)
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.evaluator = None

        self.model = model_config.model
        self.model.to(device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train model locally."""
        attack_activated = bool(config["attack_activated"])
        lr = float(config.get("lr", settings.model.learning_rate))
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
        )

        # Return locally-trained model and metrics
        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"id": self.partition_id},
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model on the local validation set."""
        set_weights(self.model, parameters)

        if settings.use_case is not None and settings.use_case.name == "AVISENCE":
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
        else:
            loss, accuracy = test(self.model, self.val_loader)
            return (
                loss,
                len(self.val_loader.dataset),
                {"accuracy": accuracy},
            )
