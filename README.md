# Trackmania RL — bot SAC qui apprend à conduire TM2020

Un agent d'apprentissage par renforcement (Soft Actor-Critic) qui apprend à
conduire dans **Trackmania 2020** et à optimiser ses temps, en temps réel.

## Comment ça marche

- **Perception** : 19 rayons LIDAR extraits d'une capture d'écran du jeu + données
  du véhicule (vitesse, position, inputs, gear, rpm), via le plugin OpenPlanet
  `TMRL_GrabData.op`. Les observations sont empilées sur 4 frames pour percevoir
  le mouvement.
- **Contrôle** : gamepad Xbox 360 virtuel (`vgamepad`) — gaz à fond, l'agent gère
  direction + frein.
- **Récompense** : progression le long d'une trajectoire de référence enregistrée
  (méthode tmrl), via `RewardFunction`. Maximiser la distance parcourue par
  seconde = minimiser le temps au tour.
- **Algo** : SAC (stable-baselines3) avec gSDE, gamma 0.995, à 20 Hz.

## Pré-requis

1. Trackmania 2020 (Steam) + [OpenPlanet](https://openplanet.dev/)
2. Le plugin `TMRL_GrabData.op` (installé par `tmrl`)
3. `pip install -r requirements.txt`

## Utilisation

```bash
# 1. (autre map que tmrl-train) enregistrer une ligne de référence
python record_reward.py

# 2. lancer l'entraînement (TM2020 ouvert, course lancée)
python trackmania_rl/train.py

# 3. suivre l'apprentissage
tensorboard --logdir=logs

# 4. regarder l'agent entraîné
python trackmania_rl/evaluate.py
```

## Structure

| Fichier | Rôle |
|---|---|
| `config.py` | hyperparamètres et chemins |
| `trackmania_rl/environment.py` | environnement Gymnasium (LIDAR + reward tmrl) |
| `trackmania_rl/train.py` | entraînement SAC |
| `trackmania_rl/evaluate.py` | jouer un modèle entraîné |
| `trackmania_rl/callbacks.py` | métriques TensorBoard |
| `record_reward.py` | enregistrer la trajectoire de référence d'une map |
| `diag.py` | diagnostic de la perception LIDAR |
