import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader, Subset

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

        from modules.utils import iouEval

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
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        :param parameters : The current (global) model parameters.
        :param config : Configuration parameters which allow the server to influence training
        on the client. It can be used to communicate arbitrary values from the server to the client,
        for example, to set the number of (local) training epochs.
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
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        :param parameters : The current (global) model parameters.
        :param config : Configuration parameters which allow the server to influence evaluation
        on the client. It can be used to communicate arbitrary values from the server to the client,
        for example, to influence the number of examples used for evaluation.
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
