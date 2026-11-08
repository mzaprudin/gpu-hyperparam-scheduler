"""
Centralized argument parsing and TrainingParameters creation.
Provides consistent parameter handling across different scripts.
"""

from gpu_hyperparam_scheduler.config import TrainingParameters
from gpu_hyperparam_scheduler.args import parse_args


def create_training_params(args=None) -> TrainingParameters:
    """
    Create TrainingParameters from command line arguments.

    Returns:
        TrainingParameters configured from command line args
    """
    args = args if args is not None else parse_args()

    return TrainingParameters(
        gpu_idx=-1 if args.cpu else args.gpu,
        name=args.name,
        data_file=args.data,
        save_dir=args.save_dir,
        test_hours=args.test_hours,
        val_hours=args.val_hours,
        train_hours=args.train_hours,
        hours_offset=args.hours_offset,
        life_hours=args.life_hours,
        model_to_load=args.model,
        batch_size=args.batch,
        num_epochs=args.epochs,
        sequence_length=args.window,
        step=args.step,
        forward_length=args.forward,
        skip_length=args.skip,
        up_more_than=args.up,
        down_less_than=args.dn,
        min_price_range_percent=args.range,
        min_liquidity=args.liquidity,
        min_trading_volume=args.volume,
        reverse_volume_filter=args.reverse,
        min_tp_ratio=args.tp,
        min_threshold=args.threshold,
        test_only=args.test,
        standard_normalizer=args.standard_normalizer,
        limit=args.limit,
        seed=args.seed,
    )

def copy_training_params(params: TrainingParameters, loaded_params: TrainingParameters):
    if loaded_params is None:
        return

    params.features = loaded_params.features
    params.step = loaded_params.step
    params.sequence_length = loaded_params.sequence_length
    params.forward_length = loaded_params.forward_length
    params.skip_length = loaded_params.skip_length
    params.test_hours = loaded_params.test_hours
    params.val_hours = loaded_params.val_hours
    params.min_liquidity = loaded_params.min_liquidity
    params.min_trading_volume = loaded_params.min_trading_volume
    params.min_price_range_percent = loaded_params.min_price_range_percent
    params.up_more_than = loaded_params.up_more_than
    params.down_less_than = loaded_params.down_less_than
    params.reverse_volume_filter = loaded_params.reverse_volume_filter
