import os
import sys

from dataset import make_dataloader
from mmgait.mmw_model.models import make_model
from mmgait.mmw_model.utils.meter import AverageMeter

current_directory = os.path.dirname(os.path.abspath("mmgait"))
sys.path.insert(0, current_directory)

import argparse
import random
import numpy as np
import torch
from utils.logger import setup_logger
from config import cfg
from torch import nn, optim



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config_file", default="config/milipoint/iden.yaml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()

    logger.info("Config")
    logger.info(cfg)

    # Dataloader
    train_loader, val_loader, test_loader, info = make_dataloader(cfg)

    # Model
    model = make_model(cfg, info)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LEARNING_RATE)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Number of epochs
    n_epochs = cfg.SOLVER.MAX_EPOCHS
    # scheduler

    # train_model
    # Training and validation
    logger.info("Starting Training.")
    loss_meter = AverageMeter()
    for epoch in range(n_epochs):
        # Training
        model.train()
        loss_meter.reset()
        for i , (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss)
            if (i+1) % cfg.SOLVER.LOG_PERIOD == 0:
                logger.info(f"Train :: Epoch : {epoch+1}/{cfg.SOLVER.MAX_EPOCHS}, Iter : {i}/{len(train_loader)}, loss : {loss_meter.avg}")

        # Validation
        model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                accuracy = (output.argmax(dim=1) == target).float().mean().item()
                acc_meter.update(accuracy)
        logger.info(f"Validation :: Epoch : {epoch + 1}/{cfg.SOLVER.MAX_EPOCHS}, accuracy : {acc_meter.avg}")

        if epoch % cfg.SOLVER.SAVE_AFTER == 0:
            torch.save(model.state_dict(), f'{cfg.OUTPUT_DIR}\model_{epoch}.pth')


    # test_model