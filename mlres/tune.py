import os
import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback
from torch.utils.data import DataLoader
from mlres.dataset import GermanWeatherEnergy
from mlres.modules.networks import WaveNet

def train_epoch(model, train_loader, optimizer, criterion, log_interval):
    model.train()
    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(torch.transpose(x, 2, 1)).reshape(y.shape)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i % log_interval == 0:
            print(f"Batch {i}, Loss: {train_loss / log_interval}")
            train_loss = 0
    
def validate_epoch(model, val_loader, criterion, log_interval):
    model.eval()
    total_val_loss, val_loss = 0, 0
    for i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            y_hat = model(torch.transpose(x, 2, 1)).reshape(y.shape)
            loss = criterion(y_hat, y)
            total_val_loss += loss.item()
            val_loss += loss.item()
            if i % log_interval == 0:
                print(f"Validation Batch {i}, Loss: {val_loss / log_interval}")
                val_loss = 0
    return total_val_loss / len(val_loader)

def objective(config, weather_energy_data_ref):
    import ray.train.torch
    weather_energy_data = weather_energy_data_ref
    
    # Loaders
    device = ray.train.torch.get_device()
    weather_energy_data.device = device
    weather_energy_data.window_size = config['window_size']
    train_data, val_data = weather_energy_data.get_train_and_val_data()
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    n_features = train_loader.dataset.data.shape[1]
    model = WaveNet(n_layers=config['n_layers'], 
                    n_input_channels=n_features, 
                    n_channels=config['n_channels'], 
                    window_size=config['window_size'], 
                    horizon=config['horizon']).to(device)
    
    # Training
    optimizer = torch.optim.NAdam(model.parameters(), lr=config['learning_rate'], weight_decay=config['reg_coeff'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])
    criterion = torch.nn.L1Loss()
    
    while True:
        train_epoch(model, train_loader, optimizer, criterion, config['log_interval'])
        val_loss = validate_epoch(model, val_loader, criterion, config['log_interval'])
        train.report({'val_loss': val_loss})
        scheduler.step()

def main():
    import ray
    ray.init()
    
    # Hyperparameters
    config = dict()
    config['horizon'] = 96
    config['log_interval'] = 100
    config['data_dir'] = "/home/martius-lab/renewable_ml/data/"

    # Dataset
    device = torch.device('cpu')
    weather_energy_data = GermanWeatherEnergy(window_size=288, horizon=config['horizon'], data_dir=config['data_dir'], device=device)
    
    # Put the dataset in Ray's object store
    weather_energy_data_ref = ray.put(weather_energy_data)
    
    # Define search space
    search_space = {
        'n_layers': tune.randint(8, 12),
        'n_channels': tune.choice([64, 128, 256, 512]),
        'batch_size': tune.choice([32, 64, 128]),
        'learning_rate': tune.loguniform(1e-6, 1e-3),
        'lr_decay': tune.uniform(0.8, 1.0),
        'window_size': tune.choice([192, 288, 384]),
        'reg_coeff': tune.choice([0.05, 0.1, 0.15]),
        **config
    }
    algo = OptunaSearch()
    
    param_wrapped = tune.with_parameters(objective, weather_energy_data_ref=weather_energy_data_ref)
    resource_wrapped = tune.with_resources(param_wrapped, resources={'gpu': 1})
    # Hyperparameter tuning
    tuner = tune.Tuner(
        resource_wrapped,
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            search_alg=algo,
            num_samples=50,
            time_budget_s= 23*60*60, 
        ),
        run_config=train.RunConfig(
            storage_path=os.path.join(config['data_dir'], '../results'),
            name='tuning_experiment_photovoltaic',
            stop={"training_iteration": 10},
            callbacks=[WandbLoggerCallback(
                project="MLRES - Photovoltaic",
                api_key=os.environ["WANDB_API_KEY"],
                log_config=True
            )]
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == "__main__":
    main()
