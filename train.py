import argparse
from bunch import Bunch
from loguru import logger
from ruamel.yaml import YAML
# from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import VesselDataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch


def main(CFG, data_path, batch_size, with_val=True):
    """
    Main training function.
    :param CFG: Configuration object from config.yaml.
    :param data_path: Path to the dataset (Spatial directory).
    :param batch_size: Batch size for training and validation.
    :param with_val: Whether to include validation during training.
    """
    seed_torch()

    # Load train dataset
    train_dataset = VesselDataset(data_path, mode="training")
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )

    # Load validation dataset if required
    val_loader = None
    if with_val:
        val_dataset = VesselDataset(data_path, mode="validation")
        val_loader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
        )

    # Logging dataset stats
    logger.info(f'Total training samples: {len(train_dataset)}')
    if with_val:
        logger.info(f'Total validation samples: {len(val_dataset)}')

    # Initialize model
    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')

    # Initialize loss
    loss = get_instance(losses, 'loss', CFG)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/lwt/data_pro/vessel/Spatial", type=str,
                        help='Path to the dataset (Spatial directory)')
    parser.add_argument('-bs', '--batch_size', default=512, type=int,
                        help='Batch size for training and validation')
    parser.add_argument("--val", help="Include validation during training",
                        required=False, default=True, action="store_true")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open('/kaggle/working/FR-UNet-RVD/config.yaml', encoding='utf-8') as file:
        # CFG = Bunch(safe_load(file))
        yaml = YAML(typ='safe', pure=True)
        CFG = Bunch(yaml.load(file))

    # Call main function
    main(CFG, args.dataset_path, args.batch_size, args.val)
