# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from matplotlib.figure import Figure
from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr, torch_utils

try:
    assert not TESTS_RUNNING  # do not log pytest
    #assert SETTINGS["tensorboard"] is True  # verify integration is enabled
    WRITER = None  # TensorBoard SummaryWriter instance
    PREFIX = colorstr("TensorBoard: ")

    # Imports below only required if TensorBoard enabled
    import warnings
    from copy import deepcopy

    import torch
    from torch.utils.tensorboard import SummaryWriter

except (ImportError, AssertionError, TypeError, AttributeError):
    # TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows
    # AttributeError: module 'tensorflow' has no attribute 'io' if 'tensorflow' not installed
    SummaryWriter = None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    """Log scalar values to TensorBoard.

    Args:
        scalars (dict): Dictionary of scalar values to log to TensorBoard. Keys are scalar names and values are the
            corresponding scalar values.
        step (int): Global step value to record with the scalar values. Used for x-axis in TensorBoard graphs.

    Examples:
        Log training metrics
        >>> metrics = {"loss": 0.5, "accuracy": 0.95}
        >>> _log_scalars(metrics, step=100)
    """
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)
            
def _log_figure(figure: Figure, step=0):
    if WRITER:
        tag: str = figure.axes[0].get_title()
        WRITER.add_figure(tag, figure, step)

def _log_tensorboard_graph(trainer) -> None:
    """Log model graph to TensorBoard.

    This function attempts to visualize the model architecture in TensorBoard by tracing the model with a dummy input
    tensor. It first tries a simple method suitable for YOLO models, and if that fails, falls back to a more complex
    approach for models like RTDETR that may require special handling.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer object containing the model to visualize. Must
            have attributes model and args with imgsz.

    Notes:
        This function requires TensorBoard integration to be enabled and the global WRITER to be initialized.
        It handles potential warnings from the PyTorch JIT tracer and attempts to gracefully handle different
        model architectures.
    """
    # Input image
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())  # for device, type
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # suppress jit trace warning
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)  # suppress jit trace warning

        # Try simple method first (YOLO)
        try:
            trainer.model.eval()  # place in .eval() mode to avoid BatchNorm statistics changes
            WRITER.add_graph(torch.jit.trace(torch_utils.unwrap_model(trainer.model), im, strict=False), [])
            LOGGER.info(f"{PREFIX}model graph visualization added ✅")
            return

        except Exception:
            # Fallback to TorchScript export steps (RTDETR)
            try:
                model = deepcopy(torch_utils.unwrap_model(trainer.model))
                model.eval()
                model = model.fuse(verbose=False)
                for m in model.modules():
                    if hasattr(m, "export"):  # Detect, RTDETRDecoder (Segment and Pose use Detect base class)
                        m.export = True
                        m.format = "torchscript"
                model(im)  # dry run
                WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
                LOGGER.info(f"{PREFIX}model graph visualization added ✅")
            except Exception as e:
                LOGGER.warning(f"{PREFIX}TensorBoard graph visualization failure {e}")

def _log_tensorboard_histogram(trainer):
    if WRITER:
        for _, (name, param) in enumerate(trainer.model.named_parameters()):
            name_parts: str = name.split('.')
            module = ".".join(name_parts[:3])
            layer = ".".join(name_parts[3:])
            WRITER.add_histogram(tag=f"{module}/{layer}", values=param, global_step=trainer.epoch + 1)
            
def on_pretrain_routine_start(trainer) -> None:
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}TensorBoard not initialized correctly, not logging this run. {e}")


def on_train_start(trainer) -> None:
    """Log TensorBoard graph."""
    if WRITER:
        _log_tensorboard_graph(trainer)

def on_train_epoch_end(trainer) -> None:
    """Log scalar statistics at the end of a training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    _log_tensorboard_histogram(trainer)


def on_fit_epoch_end(trainer) -> None:
    """Log epoch metrics at end of training epoch."""
    _log_scalars(trainer.metrics, trainer.epoch + 1)
    
    # Log class metrics - support both standard and dual detection
    if hasattr(trainer.validator, 'epoch_class_metrics'):
        for k, v in trainer.validator.epoch_class_metrics:
            cls, metric = k.split('_')
            _log_scalars({f'{metric}/{cls}': v}, trainer.epoch + 1)
    if hasattr(trainer.validator, 'epoch_class1_metrics'):
        for k, v in trainer.validator.epoch_class1_metrics:
            cls, metric = k.split('_')
            _log_scalars({f'{metric}/{cls}': v}, trainer.epoch + 1)
    if hasattr(trainer.validator, 'epoch_class2_metrics'):
        for k, v in trainer.validator.epoch_class2_metrics:
            cls, metric = k.split('_')
            _log_scalars({f'{metric}/{cls}': v}, trainer.epoch + 1)
    
    validator = trainer.validator
    
    # Log epoch figures (standard)
    if hasattr(validator, 'epoch_figures'):
        for figure in validator.epoch_figures:
            _log_figure(figure, trainer.epoch + 1)
            

def on_train_end(trainer):
    if WRITER:
        if hasattr(trainer, "validator"):
            validator = trainer.validator
            
            # Log final figures
            if hasattr(validator, 'final_figures'):
                for figure in validator.final_figures:
                    _log_figure(figure, 0)
                    
            # Log main metrics figures
            if hasattr(validator, 'metrics') and hasattr(validator.metrics, 'figures'):
                for figure in validator.metrics.figures:
                    _log_figure(figure, 0)
                    
            # Log dual class figures (dual detection support)
            if hasattr(validator, 'metrics1') and hasattr(validator.metrics1, 'figures'):
                for figure in validator.metrics1.figures:
                    _log_figure(figure, 0)
            if hasattr(validator, 'metrics2') and hasattr(validator.metrics2, 'figures'):
                for figure in validator.metrics2.figures:
                    _log_figure(figure, 0)
                    
        if hasattr(trainer, "test_validator"):
            test_validator = trainer.test_validator
            
            # Log test figures
            if hasattr(test_validator, 'final_figures'):
                for figure in test_validator.final_figures:
                    _log_figure(figure, 0)
                    
            if hasattr(test_validator, 'metrics') and hasattr(test_validator.metrics, 'figures'):
                for figure in test_validator.metrics.figures:
                    _log_figure(figure, 0)
                    
            # Log dual test class figures (dual detection support)
            if hasattr(test_validator, 'metrics1') and hasattr(test_validator.metrics1, 'figures'):
                for figure in test_validator.metrics1.figures:
                    _log_figure(figure, 0)
            if hasattr(test_validator, 'metrics2') and hasattr(test_validator.metrics2, 'figures'):
                for figure in test_validator.metrics2.figures:
                    _log_figure(figure, 0)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
    }
    if SummaryWriter
    else {}
)
