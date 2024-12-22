
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import yaml
import numpy as np
import torch
import time
import math
import pandas as pd
from model import Transformer,AttentiveFP
from dataset import SMILES_dataset,Graph_dataset
import random
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.cuda.amp import autocast
from joblib import Parallel, delayed


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(model, loader, optimizer):
    model.train()
    loss_all = []
    all_predictions = []
    all_targets = []


    with torch.set_grad_enabled(True):
        for datas,label,_ in loader:
            optimizer.zero_grad()

            if config['model_type'] == 'gnn':
                datas, label = datas.to(device), label.to(device)
                label = normalizer.norm(label)
                output = model(datas)

            elif config['model_type'] == 'tf':

                data = [data.to(device) for data in datas]

                label = label.to(device)
                label = normalizer.norm(label)
                output = model(data)


            label = torch.squeeze(normalizer.denorm(label))
            output = torch.squeeze(normalizer.denorm(output))
            loss = F.mse_loss(output, label)

            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            all_predictions.append(output.detach().cpu())
            all_targets.append(label.detach().cpu())



    all_predictions = torch.cat([pred for pred in all_predictions if pred.dim() > 0], dim=0)
    all_targets = torch.cat([target for target in all_targets if target.dim() > 0], dim=0)

    mae_all = F.l1_loss(all_predictions, all_targets).item()
    r2_all = r2_score(all_predictions.numpy(), all_targets.numpy())
    mse_all = F.mse_loss(all_predictions, all_targets).item()

    loss = np.average(loss_all)
    mae = mae_all
    r2 = r2_all
    rmse = math.sqrt(mse_all)

    return loss, mae, r2, rmse


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def evaluate_dataset(model, dataset, fold,shuffle,name, save=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

    y, pred,smi = [],[],[]


    with torch.no_grad():
        for datas, label,smiles in loader:

            if config['model_type'] == 'gnn':
                datas, label = datas.to(device), label.to(device)
                label = normalizer.norm(label)
                output = model(datas)

            elif config['model_type'] == 'tf':
                data = [data.to(device) for data in datas]
                label = label.to(device)
                label = normalizer.norm(label)
                output = model(data)


            label = normalizer.denorm(label)
            output = normalizer.denorm(output)

            y.extend(label.detach().cpu().numpy())
            pred.extend(output.detach().cpu().numpy())
            smi.extend(smiles)



    y, pred = np.array(y).flatten(), np.array(pred).flatten()
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    mse = mean_squared_error(y, pred)

    if save:
        df = pd.DataFrame({'Experimental Value': y, 'Predicted Value': pred, 'SMILES': smi})
        df.to_csv(f'training_results/cv/fold_{fold + 1}_predictions_{name}.csv', index=False)

    return mae, rmse, r2, mse



def predict_with_model(model, test_loader, means, stds, device):
    test_pred = []
    with torch.no_grad():
        for datas, _, _ in test_loader:

            datas = datas.to(device)
            # label = normalizer.norm(label)
            # output = model(datas)



            # data = [data.to(device) for data in datas]
            with autocast():
                output = model(datas)
            output = output * stds + means
            test_pred.append(output)  # 保持为张量
    return torch.cat(test_pred).cpu().numpy()  # 最后转换为 NumPy



def load_model(model_path, config, device):
    if config['model_type'] == 'gnn':
        model = AttentiveFP(**config["gnn"]).to(device)
    elif config['model_type'] == 'tf':
        model = Transformer(**config["transformer"]).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_csv(config,df,name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [f'model_weight/cv/fold_{i}_best_model.pth' for i in range(1, 6)]

    means, stds = -3.0441, 2.2845
    outputs = []

    test_dataset = SMILES_dataset(df, tokenizer=SMILES_Atomwise_Tokenizer('vocab.txt'))
    # test_dataset = Graph_dataset(df)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=False)

    # Load models in parallel
    models = Parallel(n_jobs=3)(delayed(load_model)(path, config, device) for path in model_paths)

    for model in models:
        test_pred = predict_with_model(model, test_loader, means, stds, device)
        outputs.append(test_pred)

    outputs = np.array(outputs)
    mean_outputs = np.mean(outputs, axis=0).flatten()
    y = df['LogS(mol/L)'].to_numpy().flatten()

    mae = mean_absolute_error(y, mean_outputs)
    rmse = np.sqrt(mean_squared_error(y, mean_outputs))
    r2 = r2_score(y, mean_outputs)

    df0 = pd.DataFrame({'Experimental Value': y, 'Predicted Value': mean_outputs})
    df0.to_csv(f'training_results/predictions_{name}.csv', index=False)


    return mae, rmse, r2


def calculate_metrics(csv_path):

    data = pd.read_csv(csv_path)
    experimental_values = []
    predicted_values = []

    for i in range(0, len(data), 10):
        predicted_value = data['Predicted Value'][i:i+10]
        experimental_value = data['Experimental Value'][i:i+10]

        if len(set(experimental_value)) == 1:
            experimental_value = experimental_value.sample(n=1).values[0]
        else:
            print("Error: Experimental values are not the same in this group!")
            continue

        predicted_value_mean = np.mean(predicted_value)
        experimental_values.append(experimental_value)
        predicted_values.append(predicted_value_mean)

    mae = mean_absolute_error(experimental_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(experimental_values, predicted_values))
    r2 = r2_score(experimental_values, predicted_values)
    return mae, rmse, r2



def cross_validation(dataset,config,df,num_folds):

    results = {'time': [], 'train_loss': [], 'train_r2': [], 'valid_mae': [], 'valid_rmse': [], 'valid_r2': [],'test_mae': [], 'test_rmse': [], 'test_r2': []}
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])
    indices = df[config['split by']].unique()

    for fold, (train_index, test_index) in enumerate(kf.split(indices)):

        train_indices = indices[train_index]
        test_indices = indices[test_index]
        # print(len(train_indices),len(test_index))


        train_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in train_indices])
        test_dataset = Subset(dataset, [i for i, idx in enumerate(df[config['split by']]) if idx in test_indices])

        # print("Fold:", fold + 1)
        # print("Train dataset size:", len(train_dataset))
        # print("Test dataset size:", len(test_dataset))

        model = Transformer(**config["transformer"]).to(device)
        # model = AttentiveFP(**config["gnn"]).to(device)

        # 加载预训练权重文件
        if config['Load pretrained model']== True:
            print('Load pretrained model!')
            state_dict = torch.load("sigma_pretrain/sigma_model_1295.pth", map_location=device)
            model.load_state_dict(state_dict, strict=False)

        if config['Freeze']== True:
            for param in model.roberta.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['init_lr'], weight_decay=config['weight_decay'])


        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['lr_decay_patience'],
                                                                factor=config['lr_decay_factor'], min_lr=config['min_lr'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)


        best_valid_mae = float("inf")
        best_train_loss = float('inf')
        best_train_r2=0
        best_valid_rmse=float('inf')
        best_valid_r2=0
        early_stopping_count = 0
        epoch_times = []

        for epoch in range(1, config['epochs'] + 1):
            start_time = time.time()

            model.train()
            train_loss, _, train_r2, _ = train(model, train_loader, optimizer)

            if config['DA']== False:
                model.eval()
                valid_mae, valid_rmse, valid_r2, _ = evaluate_dataset(model,test_dataset ,fold,name='val',shuffle=False)
            else:
                model.eval()
                _, _, _, _ = evaluate_dataset(model,test_dataset ,fold,name='val',shuffle=False,save=True)
                valid_mae, valid_rmse, valid_r2 = calculate_metrics(f'training_results/cv/fold_{fold + 1}_predictions_val.csv')

            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            print(f"Epoch{epoch:3d},Time:{epoch_time:.2f}s,TrainLoss:{train_loss:.6f},TrainR2:{train_r2:.6f},ValidMAE:{valid_mae:.6f},ValidRMSE:{valid_rmse:.6f},ValidR2:{valid_r2:.6f}")


            # if train_r2>0.5:

            if train_r2 > 0.5 and valid_rmse < best_valid_rmse :
                best_train_loss = train_loss
                best_train_r2 = train_r2
                best_valid_mae = valid_mae
                best_valid_rmse = valid_rmse
                best_valid_r2 = valid_r2
                torch.save(model.state_dict(), f'model_weight/cv/fold_{fold + 1}_best_model.pth')

                early_stopping_count = 0
            else:
                early_stopping_count += 1


            if early_stopping_count >= config["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            lr_scheduler.step(valid_rmse)

            # current_lr = lr_scheduler.get_last_lr()
            # print(f'Epoch {epoch}: Learning rate = {current_lr}')

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        model.load_state_dict(torch.load(f'model_weight/cv/fold_{fold+1}_best_model.pth'))
        test_mae, test_rmse, test_r2,_= evaluate_dataset(model, test_dataset, fold,shuffle=False,name='test',save=True)

        results['test_mae'].append(test_mae)
        results['test_rmse'].append(test_rmse)
        results['test_r2'].append(test_r2)

        if config['DA'] == True:
            mae, rmse, r2 = calculate_metrics(f'training_results/cv/fold_{fold + 1}_predictions_test.csv')
            print(f'Best_Epoch:{epoch - early_stopping_count}, Average epoch time: {average_epoch_time:.2f}s,Train_Loss: {best_train_loss:.6f}, Train_R2: {best_train_r2:.6f},Valid_MAE: {best_valid_mae:.6f}, Valid_RMSE: {best_valid_rmse:.6f}, Valid_R2: {best_valid_r2:.6f},Test_MAE: {mae:.6f}, Test_RMSE: {rmse:.6f}, Test_R2: {r2:.6f}')

        else:
            print(f'Best_Epoch:{epoch - early_stopping_count}, Average epoch time: {average_epoch_time:.2f}s,Train_Loss: {best_train_loss:.6f}, Train_R2: {best_train_r2:.6f},Valid_MAE: {best_valid_mae:.6f}, Valid_RMSE: {best_valid_rmse:.6f}, Valid_R2: {best_valid_r2:.6f},Test_MAE: {test_mae:.6f}, Test_RMSE: {test_rmse:.6f}, Test_R2: {test_r2:.6f}')
            pass


    final_data = pd.DataFrame(columns=['Experimental Value', 'Predicted Value'])

    if config['DA'] == True:

        for i in range(1, num_folds+1):
            filename = f"training_results/cv/fold_{i}_predictions_test.csv"
            df = pd.read_csv(filename)
            grouped_df = df.groupby(df.index // 10)
            group_data_list = []
            for _, group_data in grouped_df:
                experimental_value = group_data['Experimental Value'].iloc[0]
                mean_predicted_value = group_data['Predicted Value'].mean()
                group_data_list.append({'Experimental Value': experimental_value, 'Predicted Value': mean_predicted_value})

            final_data = pd.concat([final_data.dropna(), pd.DataFrame(group_data_list)], ignore_index=True)

        mae = mean_absolute_error(final_data['Experimental Value'], final_data['Predicted Value'])
        rmse = np.sqrt(mean_squared_error(final_data['Experimental Value'], final_data['Predicted Value']))
        r2 = r2_score(final_data['Experimental Value'], final_data['Predicted Value'])


    else:
        df = pd.DataFrame()
        for i in range(1, num_folds+1):
            filename = f"training_results/cv/fold_{i}_predictions_test.csv"
            fold_df = pd.read_csv(filename)
            df = pd.concat([df, fold_df])

        mae = mean_absolute_error(df['Experimental Value'], df['Predicted Value'])
        rmse = np.sqrt(mean_squared_error(df['Experimental Value'], df['Predicted Value']))
        r2 = r2_score(df['Experimental Value'], df['Predicted Value'])

    # print('\n')
    print('5-fold results:',mae,rmse,r2)


    mae_1,rmse_1,r2_1=predict_csv(config,pd.read_csv("data/SC1.csv"),name='SC1')
    mae_2,rmse_2,r2_2=predict_csv(config,pd.read_csv("data/SC2.csv"),name='SC2')
    mae_3,rmse_3,r2_3=predict_csv(config,pd.read_csv("data/DB.csv"),name='DB')

    print('SC1 results:',mae_1,rmse_1,r2_1)
    print('SC2 results:',mae_2,rmse_2,r2_2)
    print('DB results:',mae_3,rmse_3,r2_3)


if __name__ == "__main__":

    config = yaml.load(open("config.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)

    setup_seed(config['seed'])
    print(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on:',device)
    print('---loading dataset---')

    df = pd.read_csv("data/dataset.csv")
    from tokenizer import SMILES_Atomwise_Tokenizer
    tokenizer=SMILES_Atomwise_Tokenizer('vocab.txt')

    # dataset = Graph_dataset(df)
    dataset = SMILES_dataset(df = df, tokenizer = tokenizer)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    labels = []
    for data,label,_ in loader:
        labels.append(label)
    labels = torch.cat(labels)
    normalizer = Normalizer(labels)
    print(normalizer.mean, normalizer.std, labels.shape)


    cross_validation(dataset, config, df,config['k'])

