from config import *
from model import *
from train import *
from dataset import *


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)
    #dfx = pd.read_csv(config.TRAINING_FILE).sample(n=2000).reset_index(drop=True)
    
    pre_test = pd.read_csv('../input/tweet-make-testpse/pse_test.csv')
    pre_test.drop(['sentiment_main'], axis=1, inplace=True)
    pre_test.loc[:,'kfold']=6
    dfx = pd.concat([dfx, pre_test])

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    
    print(df_train.shape)
    print(df_valid.shape)
    
    
    
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
#     optimizer_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
#     ]
#     optimizer = AdamW(optimizer_parameters, lr=3e-5)
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if 'roberta' in n and any(nd in n for nd in no_decay)], 'lr': 3e-5, 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if 'roberta' not in n and any(nd in n for nd in no_decay)], 'lr': 3e-5 * 500, 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if 'roberta' in n and not any(nd in n for nd in no_decay)], 'lr': 3e-5 , 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if 'roberta' not in n and not any(nd in n for nd in no_decay)], 'lr': 3e-5 * 500, 'weight_decay': 0.001},
    ]
    optimizer = AdamW(optimizer_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    es = utils.EarlyStopping(patience=2, mode="max",  delta=0.0005)
    print(f"Training is Starting for fold={fold}")
    
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"model_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)