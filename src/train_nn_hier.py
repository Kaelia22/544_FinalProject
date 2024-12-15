import argparse
import dataclasses
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def contain_text(text_df, pair_id: str):
    text_id1, text_id2 = pair_id.split("_")
    return (text_id1 in text_df.text_id.values) and (text_id2 in text_df.text_id.values)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat.flatten(), y))


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = MyModel(
            model_path=cfg.MODEL_PATH,
            num_classes=cfg.NUM_CLASSES,
            transformer_params=cfg.TRANSFORMER_PARAMS,
            custom_header=cfg.CUSTOM_HEADER,
            embedding_layers=cfg.EMBEDDING_LAYERS, ## ADDED ##
        )
        self.criterion = RMSELoss()
        self.lr = cfg.LEARNING_RATE
        self.validation_outputs = []

    def forward(self, x):
        ids1 = x["ids1"]
        ids2 = x["ids2"]
        attention_mask1 = x["attention_mask1"]
        attention_mask2 = x["attention_mask2"]
        token_type_ids1 = x["token_type_ids1"]
        token_type_ids2 = x["token_type_ids2"]
        features = x["features"]
        output = self.backbone(
            input_ids1=ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            input_ids2=ids2,
            attention_mask2=attention_mask2,
            token_type_ids2=token_type_ids2,
            features=features,
        )
        return output

    def training_step(self, batch, batch_idx):
        ids1 = batch["ids1"]
        ids2 = batch["ids2"]
        attention_mask1 = batch["attention_mask1"]
        attention_mask2 = batch["attention_mask2"]
        token_type_ids1 = batch["token_type_ids1"]
        token_type_ids2 = batch["token_type_ids2"]
        features = batch["features"]
        targets = batch["targets"]
        output = self.backbone(
            input_ids1=ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            input_ids2=ids2,
            attention_mask2=attention_mask2,
            token_type_ids2=token_type_ids2,
            features=features,
        )
        loss = self.criterion(output, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        ids1 = batch["ids1"]
        ids2 = batch["ids2"]
        attention_mask1 = batch["attention_mask1"]
        attention_mask2 = batch["attention_mask2"]
        token_type_ids1 = batch["token_type_ids1"]
        token_type_ids2 = batch["token_type_ids2"]
        features = batch["features"]
        targets = batch["targets"]
        output = self.backbone(
            input_ids1=ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            input_ids2=ids2,
            attention_mask2=attention_mask2,
            token_type_ids2=token_type_ids2,
            features=features,
        )
        loss = self.criterion(output, targets)

        ## ADDED ##
        preds = torch.round(output)
        ###########

        # output = OrderedDict(
        #     {
        #         "targets": targets.detach(),
        #         "preds": output.detach(),
        #         "loss": loss.detach(),
        #     }
        # )
        self.validation_outputs.append({
            "targets": targets.detach(),
            "preds": output.detach(),
            "loss": loss.detach(),
        })
        return output

    # def validation_epoch_end(self, outputs):
    #     d = dict()
    #     d["epoch"] = int(self.current_epoch)
    #     d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

    #     targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
    #     preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()

    #     score = pd.DataFrame({"targets": targets, "preds": preds}).corr()["targets"][
    #         "preds"
    #     ]
    #     d["v_score"] = score
    #     self.log_dict(d, prog_bar=True)
    # def on_validation_epoch_end(self):
    #     d = dict()
    #     d["epoch"] = int(self.current_epoch)
        
    #     # Process stored validation outputs
    #     outputs = self.validation_outputs
    #     d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

    #     targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
    #     preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()

    #     score = pd.DataFrame({"targets": targets, "preds": preds}).corr()["targets"][
    #         "preds"
    #     ]
    #     d["v_P_score"] = score

    #     # rmse, mae, r^2
    #     rmse = np.sqrt(mean_squared_error(targets, preds))
    #     mae = mean_absolute_error(targets, preds)
    #     r2 = r2_score(targets, preds)

    #     d["v_rmse"] = rmse
    #     d["v_mae"] = mae
    #     d["v_r2"] = r2
        
    #     # Log the metrics
    #     self.log_dict(d, prog_bar=True)
        
    #     # Clear the outputs for the next epoch
    #     self.validation_outputs.clear()
    def on_validation_epoch_end(self):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        
        # Process stored validation outputs
        outputs = self.validation_outputs
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()
    
        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()
    
        # Overall metrics
        score = pd.DataFrame({"targets": targets, "preds": preds}).corr()["targets"][
            "preds"
        ]
        d["v_P_score"] = score
    
        # rmse, mae, r^2
        rmse = np.sqrt(mean_squared_error(targets, preds))
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)
    
        d["v_rmse"] = rmse
        d["v_mae"] = mae
        d["v_r2"] = r2
        
        # If it's the last epoch, calculate metrics by language-language pair
        if self.current_epoch == (self.trainer.max_epochs - 1):
            # Load the validation dataframe to get language information
            val_df = self.trainer.datamodule.valid_df
            
            # Add predictions to the dataframe
            val_df['preds'] = preds
            
            # Group by language pair and calculate metrics
            language_pair_metrics = val_df.groupby('meta_lang').apply(lambda group: {
                'v_P_score': pd.DataFrame({
                    'targets': group['Overall'], 
                    'preds': group['preds']
                }).corr()['targets']['preds'],
                'v_rmse': np.sqrt(mean_squared_error(group['Overall'], group['preds'])),
                'v_mae': mean_absolute_error(group['Overall'], group['preds']),
                'v_r2': r2_score(group['Overall'], group['preds'])
            }).to_dict()
            
            # Log metrics for each language pair
            for lang_pair, metrics in language_pair_metrics.items():
                d[f"{lang_pair}_v_P_score"] = metrics['v_P_score']
                d[f"{lang_pair}_v_rmse"] = metrics['v_rmse']
                d[f"{lang_pair}_v_mae"] = metrics['v_mae']
                d[f"{lang_pair}_v_r2"] = metrics['v_r2']
        
        # Log the metrics
        self.log_dict(d, prog_bar=True)
        
        # Clear the outputs for the next epoch
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class TextDataset(Dataset):
    def __init__(
        self,
        df,
        text_col: str,
        target_col: str,
        tokenizer_name: str,
        max_len: int,
        is_train: bool = True,
    ):
        super().__init__()

        self.df = df
        self.is_train = is_train

        if self.is_train:
            self.target = torch.tensor(self.df[target_col].values, dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoded1 = tokenizer.batch_encode_plus(
            self.df[f"{text_col}_1"].tolist(),
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        self.encoded2 = tokenizer.batch_encode_plus(
            self.df[f"{text_col}_2"].tolist(),
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        self.features = (
            pd.read_csv("544_FinalProject/input/semeval2022/X_train.csv")
            if is_train
            else pd.read_csv("544_FinalProject/input/semeval2022/X_test.csv")
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids1 = torch.tensor(self.encoded1["input_ids"][index])
        attention_mask1 = torch.tensor(self.encoded1["attention_mask"][index])
        token_type_ids1 = torch.tensor(self.encoded1["token_type_ids"][index])
        input_ids2 = torch.tensor(self.encoded2["input_ids"][index])
        attention_mask2 = torch.tensor(self.encoded2["attention_mask"][index])
        token_type_ids2 = torch.tensor(self.encoded2["token_type_ids"][index])
        features = torch.tensor(self.features.loc[index].values, dtype=torch.float32)
        if self.is_train:
            target = self.target[index]
            return {
                "ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "token_type_ids1": token_type_ids1,
                "ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "token_type_ids2": token_type_ids2,
                "features": features,
                "targets": target,
            }
        else:
            return {
                "ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "token_type_ids1": token_type_ids1,
                "ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "token_type_ids2": token_type_ids2,
                "features": features,
            }


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.test_df = None
        self.train_df = None
        self.valid_df = None
        self.cfg = cfg

    # def merge_df_and_text(self, df, text_dataframe):
    #     text_dataframe = text_dataframe.dropna(subset=["title", "text"]).reset_index(
    #         drop=True
    #     )
    #     text_dataframe["title"] = (
    #         text_dataframe["title"].fillna("")
    #         + "[SEP]"
    #         + text_dataframe["text"].fillna("")
    #     )
    #     text_dataframe["text_id"] = text_dataframe["text_id"].astype(str)
    #     df = df[
    #         df["pair_id"].map(lambda x: contain_text(text_dataframe, x))
    #     ].reset_index(drop=True)
    #     df["text_id1"] = df["pair_id"].str.split("_").map(lambda x: x[0])
    #     df["text_id2"] = df["pair_id"].str.split("_").map(lambda x: x[1])

    #     df = pd.merge(
    #         df,
    #         text_dataframe[["text_id", "title"]],
    #         left_on="text_id1",
    #         right_on="text_id",
    #         how="left",
    #     )
    #     df = pd.merge(
    #         df,
    #         text_dataframe[["text_id", "title"]],
    #         left_on="text_id2",
    #         right_on="text_id",
    #         how="left",
    #         suffixes=("_1", "_2"),
    #     )
    #     return df
    def merge_df_and_text(self, df, text_dataframe):
        text_dataframe = text_dataframe.dropna(subset=["title", "text"]).reset_index(
            drop=True
        )
        if 'meta_lang' not in text_dataframe.columns:
            print("Warning: 'meta_lang' column not found in text_dataframe")
            
        text_dataframe["title"] = (
            text_dataframe["title"].fillna("")
            + "[SEP]"
            + text_dataframe["text"].fillna("")
        )
        text_dataframe["text_id"] = text_dataframe["text_id"].astype(str)
        df = df[
            df["pair_id"].map(lambda x: contain_text(text_dataframe, x))
        ].reset_index(drop=True)
        df["text_id1"] = df["pair_id"].str.split("_").map(lambda x: x[0])
        df["text_id2"] = df["pair_id"].str.split("_").map(lambda x: x[1])

        # df = pd.merge(
        #     df,
        #     text_dataframe[["text_id", "title"]],
        #     left_on="text_id1",
        #     right_on="text_id",
        #     how="left",
        # )
        # df = pd.merge(
        #     df,
        #     text_dataframe[["text_id", "title"]],
        #     left_on="text_id2",
        #     right_on="text_id",
        #     how="left",
        #     suffixes=("_1", "_2"),
        # )
        if 'meta_lang' in text_dataframe.columns:
            df = pd.merge(
                df,
                text_dataframe[["text_id", "title", "meta_lang"]],
                left_on="text_id1",
                right_on="text_id",
                how="left",
            )
            df = pd.merge(
                df,
                text_dataframe[["text_id", "title", "meta_lang"]],
                left_on="text_id2",
                right_on="text_id",
                how="left",
                suffixes=("_1", "_2"),
            )
            
            # Create a combined meta_lang column
            df['meta_lang'] = df['meta_lang_1'] + '_' + df['meta_lang_2']
        return df

    def get_test_df(self):
        df = pd.read_csv(self.cfg.TEST_DF_PATH)
        text_dataframe = pd.read_csv(
            "544_FinalProject/input/semeval2022/text_dataframe_eval.csv", low_memory=False
        )
        df = self.merge_df_and_text(df, text_dataframe)
        return df

    def split_train_valid_df(self):
        if int(self.cfg.debug):
            df = pd.read_csv(self.cfg.TRAIN_DF_PATH, nrows=100)
        else:
            df = pd.read_csv(self.cfg.TRAIN_DF_PATH)

        text_dataframe = pd.read_csv(
            "544_FinalProject/input/semeval2022/text_dataframe.csv", low_memory=False
        )
        df = self.merge_df_and_text(df, text_dataframe)
        cv = KFold(n_splits=self.cfg.NUM_FOLDS, shuffle=True, random_state=42)
        for n, (train_index, val_index) in enumerate(cv.split(df)):
            df.loc[val_index, "fold"] = int(n)
        df["fold"] = df["fold"].astype(int)

        train_df = df[df["fold"] != self.cfg.fold].reset_index(drop=True)
        valid_df = df[df["fold"] == self.cfg.fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        self.test_df = self.get_test_df()
        train_df, valid_df = self.split_train_valid_df()
        self.train_df = train_df
        self.valid_df = valid_df

    def get_dataframe(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_ds(self, phase):
        assert phase in {"train", "valid", "test"}
        ds = TextDataset(
            df=self.get_dataframe(phase=phase),
            text_col=self.cfg.TEXT_COL,
            target_col=self.cfg.TARGET_COL,
            tokenizer_name=self.cfg.TOKENIZER_PATH,
            max_len=self.cfg.MAX_LEN,
            is_train=(phase != "test"),
        )
        return ds

    def get_loader(self, phase):
        dataset = self.get_ds(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=self.cfg.NUM_WORKERS,
            drop_last=(phase == "train"),
        )

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return self.get_loader(phase="train")

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


class MyModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: List[int],
        transformer_params: Dict[str, Any] = {},
        custom_header: str = "hierarchical",
        embedding_layers: List[int] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.update(transformer_params)
        self.net1 = AutoModel.from_pretrained(model_path, config=model_config)
        self.net2 = AutoModel.from_pretrained(model_path, config=model_config)
        self.out_shape = model_config.hidden_size
        self.custom_header = custom_header
        
        ## CHANGED ##
        if embedding_layers is None:
            embedding_layers = [-1, -2, -3, -4] # Default last 4 layers for hierarchical embeddings

        self.embedding_layers = embedding_layers

        if self.custom_header == "hierarchical":
            self.fc = nn.Linear(self.out_shape * 2 * len(self.embedding_layers) + 12, num_classes)
        else:
            self.fc = nn.Linear(self.out_shape * 2 + 12, num_classes)
        #############
            
    ## ADDED ##
    def extract_hierarchical_embeddings(self, outputs):
        """
        Extract and concatenate embeddings from specified layers.
        """

        hidden_states = outputs["hidden_states"]
        embeddings = torch.cat([hidden_states[layer][:, 0, :] for layer in self.embedding_layers], dim=-1)
        
        return embeddings
    ###########

    def forward(
        self,
        input_ids1,
        attention_mask1,
        token_type_ids1,
        input_ids2,
        attention_mask2,
        token_type_ids2,
        features,
    ):
        outputs1 = self.net1(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            output_hidden_states = True ## ADDED ##
        )
        outputs2 = self.net1(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            output_hidden_states = True ## ADDED ##
        )
        ## CHANGED ##

        if self.custom_header == "hierarchical":
            sequence_output1 = self.extract_hierarchical_embeddings(outputs1)
            sequence_output2 = self.extract_hierarchical_embeddings(outputs2)
        else:
            # Fallback to default max pooling
            sequence_output1, _ = outputs1["last_hidden_state"].max(1)
            sequence_output2, _ = outputs2["last_hidden_state"].max(1)
        
        sequence_output = torch.cat(
            [
                torch.abs(sequence_output1 - sequence_output2),
                sequence_output1 * sequence_output2,
            ],
            dim=1,
        )
        outputs = self.fc(torch.cat([sequence_output, features], dim=1))
        
        #############
        return outputs


@dataclasses.dataclass
class Cfg:
    PROJECT_NAME = "semeval2022"
    RUN_NAME = "exp000"
    NUM_CLASSES = 1
    NUM_EPOCHS = 5
    NUM_WORKERS = 8
    NUM_GPUS = 1
    MAX_LEN = 512
    BATCH_SIZE = 4
    TRANSFORMER_PARAMS = {
        "output_hidden_states": True,
        "hidden_dropout_prob": 0.0,
        "layer_norm_eps": 1e-7,
    }
    OUTPUT_PATH = "."
    TRAIN_DF_PATH = "544_FinalProject/input/semeval2022/semeval-2022_task8_train-data_batch.csv"
    TEST_DF_PATH = "544_FinalProject/input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    TEXT_COL = "title"
    TARGET_COL = "Overall"
    ## ADDED ##
    EMBEDDING_LAYERS = [-1, -2, -3, -4]  # Layers to use for hierarchical embedding
    ###########

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default=0)
    parser.add_argument("--max_len", default=512)
    parser.add_argument("--num_folds", default=5)
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    # parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--custom_header", default="concat")
    parser.add_argument("--lr", default=1e-5)
    args = parser.parse_args()

    debug = False
    cfg = Cfg()
    cfg.fold = int(args.fold)
    cfg.debug = debug
    cfg.MAX_LEN = int(args.max_len)
    cfg.NUM_FOLDS = int(args.num_folds)
    cfg.MODEL_PATH = args.model
    cfg.TOKENIZER_PATH = args.model
    cfg.CUSTOM_HEADER = args.custom_header
    cfg.LEARNING_RATE = float(args.lr)

    seed_everything(777)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # if "google.colab" in sys.modules:
    #     secret_value = "YOUR_SECRET"
    # else:
    #     from kaggle_secrets import UserSecretsClient

    #     user_secrets = UserSecretsClient()
    #     secret_value = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key='252ce179feb7d7fc79bb9100d97815ed8578542f')

    logger = CSVLogger(save_dir=str(cfg.OUTPUT_PATH), name=f"fold_{cfg.fold}")
    wandb_logger = WandbLogger(
        name=f"{cfg.RUN_NAME}_{cfg.fold}", project=cfg.PROJECT_NAME
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(cfg.OUTPUT_PATH),
        filename=f"{cfg.RUN_NAME}_fold_{cfg.fold}",
        save_weights_only=True,
        monitor=None,
    )
    
    trainer = Trainer(
    max_epochs=cfg.NUM_EPOCHS,
    accelerator="gpu" if cfg.NUM_GPUS > 0 else "cpu",  # Dynamically set the accelerator
    devices=cfg.NUM_GPUS if cfg.NUM_GPUS > 0 else None,  # Set number of devices
    callbacks=[checkpoint_callback],
    logger=[logger, wandb_logger],
    )

    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)
    trainer.fit(model, datamodule=datamodule)

    y_val_pred = torch.cat(trainer.predict(model, datamodule.val_dataloader()))
    y_test_pred = torch.cat(trainer.predict(model, datamodule.test_dataloader()))
    np.save(f"y_val_pred_fold{cfg.fold}", y_val_pred.to("cpu").detach().numpy())
    np.save(f"y_test_pred_fold{cfg.fold}", y_test_pred.to("cpu").detach().numpy())

    rule_based_pair_ids = [
        "1489951217_1489983888",
        "1615462021_1614797257",
        "1556817289_1583857471",
        "1485350427_1486534258",
        "1517231070_1551671513",
        "1533559316_1543388429",
        "1626509167_1626408793",
        "1494757467_1495382175",
    ]

    y_val_pred = np.load(f"y_val_pred_fold{cfg.fold}.npy")
    y_test_pred = np.load(f"y_test_pred_fold{cfg.fold}.npy")

    oof = datamodule.valid_df[["pair_id", cfg.TARGET_COL]].copy()
    oof["y_pred"] = y_val_pred.reshape(-1)
    oof.to_csv(f"oof_fold{cfg.fold}.csv", index=False)

    sub = pd.read_csv(cfg.TEST_DF_PATH)
    sub[cfg.TARGET_COL] = np.nan
    sub.loc[sub["pair_id"].isin(rule_based_pair_ids), cfg.TARGET_COL] = 2.8
    sub.loc[
        ~sub["pair_id"].isin(rule_based_pair_ids), cfg.TARGET_COL
    ] = y_test_pred.reshape(-1)
    
    # Because the labels of training data are reversed at the initial release
    sub["Overall"] = sub["Overall"] * -1
    sub[["pair_id", cfg.TARGET_COL]].to_csv("submission.csv", index=False)
    sub[["pair_id", cfg.TARGET_COL]].head(2)
