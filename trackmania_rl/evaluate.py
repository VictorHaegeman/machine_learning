"""
Regarder l'agent entraîné conduire dans Trackmania.

Usage
-----
  python trackmania_rl/evaluate.py                                  # modèle final
  python trackmania_rl/evaluate.py --model models/sac_trackmania_50000_steps.zip
  python trackmania_rl/evaluate.py --episodes 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os

from stable_baselines3 import SAC

import config
from trackmania_rl.environment import TrackmaniaEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def evaluate(model_path: str, n_episodes: int = 10) -> None:
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Entraîne d'abord avec :  python trackmania_rl/train.py"
        )

    env = TrackmaniaEnv()
    model = SAC.load(model_path, env=env)
    log.info("Modèle chargé : %s", model_path)

    results = []
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        total_progress = 0.0
        steps = 0
        last_info: dict = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_progress += info.get("progress_m", 0.0)
            steps += 1
            last_info = info
            done = terminated or truncated

        finished = last_info.get("finished", False)
        stuck = last_info.get("stuck", False)
        status = "TERMINÉ" if finished else ("BLOQUÉ" if stuck else "TIMEOUT")
        log.info(
            "Épisode %2d/%d | steps=%4d | reward=%8.1f | distance=%6.1f m | %s",
            ep, n_episodes, steps, total_reward, total_progress, status,
        )
        results.append({"reward": total_reward, "progress": total_progress, "finished": finished})

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_prog = sum(r["progress"] for r in results) / len(results)
    finish_rate = sum(r["finished"] for r in results) / len(results)
    log.info(
        "─── Bilan sur %d épisodes ───  reward moy=%.1f  distance moy=%.1f m  taux fin=%.0f%%",
        n_episodes, avg_reward, avg_prog, finish_rate * 100,
    )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.join(config.MODELS_DIR, f"{config.MODEL_NAME}_final"),
        metavar="PATH",
        help="Chemin du modèle .zip (défaut : models/sac_trackmania_final).",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'épisodes.")
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
