import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Entraînement SAC — agent Trackmania 2020.

SAC (Soft Actor-Critic) est l'algorithme utilisé par tous les bots TM sérieux :
- Beaucoup plus efficace que PPO sur les actions continues
- Replay buffer → réutilise les anciennes expériences
- Entropy tuning automatique → bon équilibre exploration/exploitation

Lancer
------
  python trackmania_rl/train.py

Reprendre depuis un checkpoint
-------------------------------
  python trackmania_rl/train.py --resume models/sac_trackmania_10000_steps.zip

Suivre en temps réel
--------------------
  tensorboard --logdir=logs   →  http://localhost:6006
"""

import argparse
import logging
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import config
from trackmania_rl.callbacks import TrackmaniaCallback
from trackmania_rl.environment import TrackmaniaEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def main(resume: str | None = None) -> None:
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device PyTorch : %s", device)

    log.info("Création de l'environnement — TM2020 doit être ouvert avec une course active.")
    env = Monitor(TrackmaniaEnv())

    if resume:
        log.info("Reprise depuis %s", resume)
        model = SAC.load(resume, env=env, device=device)
        # L'entropie auto s'effondre à 0 → politique figée qui retape le même mur.
        # On la FIXE pour garantir une exploration permanente et sortir de l'ornière.
        if config.RESUME_ENT_COEF is not None:
            model.ent_coef_optimizer = None
            model.log_ent_coef = None
            model.ent_coef_tensor = torch.tensor(float(config.RESUME_ENT_COEF), device=model.device)
            log.info("Entropie fixée à %.3f → exploration relancée.", config.RESUME_ENT_COEF)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            buffer_size=config.BUFFER_SIZE,
            learning_starts=config.LEARNING_STARTS,
            batch_size=config.BATCH_SIZE,
            tau=config.TAU,
            gamma=config.GAMMA,
            train_freq=config.TRAIN_FREQ,
            gradient_steps=config.GRADIENT_STEPS,
            ent_coef=config.ENT_COEF,
            target_entropy=config.TARGET_ENTROPY,
            use_sde=config.USE_SDE,
            sde_sample_freq=config.SDE_SAMPLE_FREQ,
            policy_kwargs=dict(net_arch=config.NET_ARCH),
            device=device,
            verbose=1,
            tensorboard_log=config.LOGS_DIR,
        )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config.MODEL_SAVE_FREQ,
            save_path=config.MODELS_DIR,
            name_prefix=config.MODEL_NAME,
            verbose=1,
        ),
        TrackmaniaCallback(),
    ])

    log.info("Démarrage entraînement — %d steps totaux.", config.TOTAL_TIMESTEPS)
    log.info("Ctrl+C pour arrêter proprement (le modèle sera sauvegardé).")
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            tb_log_name=config.MODEL_NAME,
            reset_num_timesteps=resume is None,
        )
    except KeyboardInterrupt:
        log.info("Arrêt demandé — sauvegarde en cours …")

    final_path = os.path.join(config.MODELS_DIR, f"{config.MODEL_NAME}_final")
    model.save(final_path)
    log.info("Modèle sauvegardé → %s.zip", final_path)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, metavar="PATH")
    args = parser.parse_args()
    main(resume=args.resume)
