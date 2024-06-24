import os
import sys
import time
import yaml
import torch
import hashlib
import numpy as np
from torch.utils.data import DataLoader
from mlres.dataset import GermanWeatherEnergy
from mlres.modules.networks import WaveNet
from tqdm import tqdm

def load_cfg(path):
    with open(path, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg

if __name__ == "__main__":
    # LOAD CONFIGURATION
    cfg = load_cfg(sys.argv[1])
    cfg_model = cfg['model']
    cfg_paths = cfg['paths']
    cfg_train = cfg['training']
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    ## PREPARATION
    if not os.path.exists(cfg_paths['results_dir']): os.makedirs(cfg_paths['results_dir'])
    name_run = str(cfg['seed']) + '_' + str(hashlib.md5(str(time.time()).encode('utf-8')).hexdigest())
    os.makedirs(os.path.join(cfg_paths['results_dir'], name_run))

    ## DATASET
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weather_energy_data = GermanWeatherEnergy(target_idx=cfg_train['target_idx'], 
                                              window_size=cfg_model['window_size'], 
                                              horizon=cfg_model['horizon'], 
                                              data_dir=cfg_paths['data_dir'], 
                                              device=device)
    # Load the splits
    train_data, val_data = weather_energy_data.get_train_and_val_data()
    test_data = weather_energy_data.get_test_data()
    # Create the dataloaders
    train_loader = DataLoader(train_data, batch_size=cfg_train['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg_train['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg_train['batch_size'], shuffle=False)

    ## MODEL
    n_features = train_data.data.shape[1]
    if cfg_model['name'] == 'WaveNet':
        model = WaveNet(n_layers=cfg_model['n_layers'], 
                        n_input_channels=n_features, 
                        n_channels=cfg_model['n_channels'], 
                        window_size=cfg_model['window_size'], 
                        horizon=cfg_model['horizon']).to(device)
    else:
        raise ValueError(f"Model {cfg_model['name']} not implemented.")

    ## TRAINING
    optimizer = torch.optim.NAdam(model.parameters(), lr=cfg_train['learning_rate'], weight_decay=cfg_train['reg_coeff'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg_train['lr_decay'])
    criterion = torch.nn.L1Loss()
    best_val_loss = float('inf')
    train_loss = 0
    n_updates = 0
    train_losses = []
    val_losses = []
    # Training loop
    for epoch in tqdm(range(cfg_train['n_epochs'])):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(torch.transpose(x, 2, 1)).reshape(y.shape)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_updates += 1
            if n_updates % cfg_train['log_interval'] == 0:
                print(f"Epoch {epoch}, Batch {i+ 1}, Loss: {train_loss / cfg_train['log_interval']}")
                train_losses.append(train_loss / cfg_train['log_interval'])
                train_loss = 0
            if n_updates % cfg_train['save_interval'] == 0:
                torch.save(model.state_dict(), os.path.join(cfg_paths['results_dir'], name_run, f"model_{n_updates}.pth"))
                print("Model saved.")
            # Validation
            if n_updates % cfg_train['validate_every_n_updates'] == 0:
                model.eval()
                total_val_loss, val_loss = 0, 0
                for i, (x, y) in enumerate(val_loader):
                    with torch.no_grad():
                        y_hat = model(torch.transpose(x, 2, 1)).reshape(y.shape)
                        loss = criterion(y_hat, y)
                        total_val_loss += loss.item()
                        val_loss += loss.item()
                        if (i + 1) % cfg_train['log_interval'] == 0:
                            print(f"Validation Batch {i + 1}, Loss: {val_loss / cfg_train['log_interval']}")
                            val_loss = 0
                val_loss = total_val_loss / len(val_loader)
                val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(cfg_paths['results_dir'], name_run, "best_model.pth"))
                    print("Model saved.")
                model.train()
        # Update learning rate
        scheduler.step()
    # Save the final model
    torch.save(model.state_dict(), os.path.join(cfg_paths['results_dir'], name_run, "final_model.pth"))
    # Save the losses
    np.save(os.path.join(cfg_paths['results_dir'], name_run, "train_losses.npy"), train_losses)
    np.save(os.path.join(cfg_paths['results_dir'], name_run, "val_losses.npy"), val_losses)
