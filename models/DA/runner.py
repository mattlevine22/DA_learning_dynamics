# Import deep learning modules
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import wandb
from pytorch_lightning.tuner import Tuner

# Import custom modules
from datasets import DynamicsDataModule, load_dyn_sys_class
from models.DA.DA_lightning import DataAssimilatorModule

class Runner:
    def __init__(self,
            seed=0,
            fast_dev_run=False,
            accelerator='auto',
            devices='auto',
            deterministic=True, # set to False to get different results each time
            project_name="l63",
            n_trajectories_train=10,
            n_trajectories_val=2,
            n_trajectories_test=2,
            T=100,
            T_long=1000,
            batch_length_T=10,
            train_sample_rate=0.01,
            test_sample_rates=[0.01],
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            batch_size=64,
            tune_batch_size=False,
            normalizer='inactive',
            dyn_sys_name='Lorenz63',
            obs_noise_std=1,
            ode_params={},
            monitor_metric='loss/val/mse',
            lr_scheduler_params={'patience': 2, 'factor': 0.1},
            tune_initial_lr=False,
            dim_state=3,
            dim_obs=1,
            use_physics=False,
            use_nn=True,
            learn_h=False,
            learn_K=False,
            init_K='hT',
            odeint_use_adjoint=False,
            odeint_method='dopri5',
            odeint_rtol=1e-7,
            odeint_atol=1e-9,
            odeint_options={'dtype': torch.float32},
            layer_width=50,
            num_hidden_layers=1,
            burnin_frac=0.75,
            learning_rate=0.001,
            dropout=0.01,
            activations='gelu',
            max_epochs=1,
            shuffle='once',
            log_every_n_steps=1,
            gradient_clip_val=10.0,
            gradient_clip_algorithm="value",
            overfit_batches=0.0):

        seed_everything(seed, workers=True)

        if devices != 'auto':
            devices = [int(devices)] # use this to specify a single device

        self.project_name = project_name

        self.data_hyperparams = {'size': {'train': n_trajectories_train,
                                                    'val': n_trajectories_val,
                                                    'test': n_trajectories_test,
                                                    },
                                 'T': {'train': T,
                                             'val': T,
                                             'test': T,},
                                 'train_sample_rate': train_sample_rate,
                                 'test_sample_rates': test_sample_rates,
                                 'batch_size': batch_size,
                                 'batch_length_T': batch_length_T, # this is the number of time steps in each batch
                                 'tune_batch_size': tune_batch_size,
                                 'dyn_sys_name': dyn_sys_name,
                                 'shuffle': shuffle,
                                 'normalizer': normalizer,
                                 'obs_noise_std': obs_noise_std, # this is the standard deviation of the observation noise
                                 'ode_params': ode_params,
                                 }

        self.model_hyperparams = {
                                  'monitor_metric': monitor_metric,
                                  'lr_scheduler_params': lr_scheduler_params,
                                  'dim_state': dim_state,
                                  'dim_obs': dim_obs,
                                  'use_physics': use_physics,
                                  'use_nn': use_nn,
                                  'learn_h': learn_h,
                                  'learn_K': learn_K,
                                  'init_K': init_K,
                                  'layer_width': layer_width,
                                  'num_hidden_layers': num_hidden_layers,
                                  'burnin_frac': burnin_frac,
                                  'learning_rate': learning_rate,
                                  'dropout': dropout,
                                  'activations': activations,
                                  'odeint_use_adjoint': odeint_use_adjoint,
                                  'odeint_method': odeint_method,
                                  'odeint_rtol': odeint_rtol,
                                  'odeint_atol': odeint_atol,
                                  'odeint_options': odeint_options,
                                  'T_long': T_long,
                                  }
        
        self.trainer_hyperparams = {'max_epochs': max_epochs,
                                    'log_every_n_steps': log_every_n_steps,
                                    'gradient_clip_val': gradient_clip_val,
                                    'gradient_clip_algorithm': gradient_clip_algorithm,
                                    'overfit_batches': overfit_batches,
                                    'deterministic': deterministic,
                                    'fast_dev_run': fast_dev_run,
                                    'limit_train_batches': limit_train_batches,
                                    'limit_val_batches': limit_val_batches,
                                    'limit_test_batches': limit_test_batches,
                                    'accelerator': accelerator,
                                    'devices': devices,
                                    }
        
        self.other_hyperparams = {'seed': seed, 'tune_initial_lr': tune_initial_lr,
                                  'accelerator': accelerator,
                                  }

        self.run()

    def run(self):
        # Combine run settings that will be logged to wandb.
        list_of_dicts = [self.other_hyperparams,
                         self.data_hyperparams,
                         self.model_hyperparams,
                         self.trainer_hyperparams]
        all_param_dict = {k: v for d in list_of_dicts for k, v in d.items()}

        # Initialize WandB logger
        wandb.init(
            project=self.project_name, config=all_param_dict)
        wandb_logger = WandbLogger()

        # Load the DataModule
        datamodule = DynamicsDataModule(**self.data_hyperparams)

        # Load the true ODE
        self.model_hyperparams['ode'] = load_dyn_sys_class(self.data_hyperparams['dyn_sys_name'])()

        # Initialize the model
        model = DataAssimilatorModule(**self.model_hyperparams)

        # Set callbacks for trainer (lr monitor, early stopping)

        # Create a PyTorch Lightning trainer with the WandbLogger
        # used this link to find LRmonitor: https://community.wandb.ai/t/how-to-log-the-learning-rate-with-pytorch-lightning-when-using-a-scheduler/3964/5
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Create an early stopping callback
        early_stopping = EarlyStopping(monitor='loss/val/mse', patience=20, mode='min', verbose=True)

        # aggregate all callbacks
        callbacks = [lr_monitor, early_stopping]


        # Initialize the trainer
        trainer = Trainer(logger=wandb_logger, callbacks=callbacks,
            # if you are running M1 Mac, the accelerator is 'mps' and will be detected automatically
            # however, pytorch nextafter does not yet support mps, so you will need to use cpu
            # can monitor this issue here (was reported Sep 2023)...hopefully updating pytorch will eventually solve this problem:
            # https://github.com/pytorch/pytorch/issues/77764
                        # other settings
                        **self.trainer_hyperparams)

        # Tune the model
        tuner = Tuner(trainer)

        # Tune the batch size
        # half the identified batch size to avoid maxing out RAM
        if self.data_hyperparams['tune_batch_size']:
            if torch.cuda.is_available():
                print('Using GPU, so setting batch size scaler max_trials to 25 (pick a smaller number if this destroys the machine)')
                max_trials = 25
            else:
                print('Using CPU, so setting batch size scaler max_trials to 6 (avoid maxing RAM on a local machine)')
                max_trials = 6
            tuner.scale_batch_size(model, max_trials=max_trials, datamodule=datamodule)
            datamodule.batch_size = max(1, datamodule.batch_size // 4)
            print('Using batch size: ', datamodule.batch_size)

        # Tune the learning rate
        if self.other_hyperparams['tune_initial_lr']:
            min_lr, max_lr = model.learning_rate * 0.01, model.learning_rate * 100
            tuner.lr_find(model, datamodule=datamodule, min_lr=min_lr, max_lr=max_lr, num_training=20)
            print('Using learning rate: ', model.learning_rate)

        # Train the model
        trainer.fit(model, datamodule=datamodule)

        # Test the model
        trainer.test(model, datamodule=datamodule)
