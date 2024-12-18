import os
import wandb
import argparse
from clearml import Task
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback

task = Task.init(project_name='Mentor Group S/Group 1', # NB: Replace YourName with your own name
                    task_name='ExperimentJ1')

task.set_base_docker('deanis/2023y2b-rl:latest')

task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

env = OT2Env(render=False)

run = wandb.init(project="sb3_RL_OT2",sync_tensorboard=True)

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

wandb_callback = WandbCallback(model_save_freq=10000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

best_model_callback = EvalCallback(
    env,  # This should be your evaluation environment
    best_model_save_path=f"models/{run.id}",
    log_path=f"runs/{run.id}",
    eval_freq=10000,
    deterministic=True,
    render=False,
    verbose=1,
    warn=False,
)

time_steps = 2500000
model.tensorboard_log = f"runs/{run.id}"
model.learn(total_timesteps=time_steps, callback=[wandb_callback, best_model_callback], progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
