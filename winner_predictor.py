import pandas as pd
from models.LogisticCatModelOneHot import LogisticCatModelOneHot
from models.LogisticCatModel import LogisticCatModel
from models.LogisticCatModelSmall import LogisticCatModelSmall
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import prepare_dataset, prepare_tensors, create_logger,  prepare_split, create_callbacks, prepare_dataLoaders, create_trainer


if __name__ == "__main__":

    model_type = "onehot"
    mode = "test"
    checkpoint_path = "./lightning_logs/sf6analisys_/Final_onehot/checkpoints/epoch=27-step=342776.ckpt"
    torch.set_float32_matmul_precision('high')

    df = pd.read_csv('./data.csv')
    df_long = pd.read_csv('./data_long.csv')
    all_chars = [
        'Jamie', 'Terry', 'Zangief', 'Kimberly', 'A.K.I.', 'Edmond Honda',
        'Ken', 'Dee Jay', 'Ryu', 'Manon', 'Marisa', 'Mai', 'Ed', 'Cammy',
        'Akuma', 'Lily', 'Luke', 'JP', 'Blanka', 'Juri', 'M. Bison',
        'Dhalsim', 'Guile', 'Chun-Li', 'Random', 'Rashid'
    ]
    num_chars = len(all_chars)

    df = prepare_dataset(df, df_long)

    X, y = prepare_tensors(df, all_chars, model_type=model_type)

    print(X.shape, y.shape)
    print("-------------------------")

    train_dataset, val_dataset, test_dataset = prepare_split(X, y)
    train_dataset, val_dataset, test_dataset = prepare_split(X, y)

# test_dataset è un Subset
    test_indices = test_dataset.indices
    torch.save(test_indices, "test_indices.pt")

    # callbacks = create_callbacks()

    # train_loader, val_loader, test_loader = prepare_dataLoaders(train_dataset, val_dataset, test_dataset)

    # logger = create_logger()
    # trainer = create_trainer(callbacks=callbacks, logger=logger)

    
    # if mode == "train":
    #     model = LogisticCatModelOneHot(
    #         num_chars=num_chars,
    #         lr=1e-2,
    #     )
    #     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # if mode == "test":
    #     best_model = LogisticCatModelOneHot.load_model_from_checkpoint(checkpoint_path)
    #     trainer.test(best_model, dataloaders=test_loader)
        
