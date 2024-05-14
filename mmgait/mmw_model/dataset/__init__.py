from mmgait.mmw_model.dataset.mmrnet_data import get_milipoint_dataset

def make_dataloader(cfg):
    if cfg.DATASET.NAME == 'milipoint':
        dataset_config = {
            'seed': 20,
            'train_split': cfg.DATASET.TRAIN_SPLIT,
            'val_split': cfg.DATASET.VAL_SPLIT,
            'test_split': cfg.DATASET.TEST_SPLIT,
            'stacks': cfg.DATASET.SEQ_LENGTH,
            'zero_padding': 'per_data_point'
        }
        loaders = get_milipoint_dataset(
            'mmr_iden',
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            workers=0,
            mmr_dataset_config=dataset_config)
        train_loader, val_loader, test_loader, info = loaders

        return train_loader, val_loader, test_loader, info