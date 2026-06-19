"""
Trackmania 2020 — environnement Gymnasium pour SAC (méthode tmrl).

Récompense : progression le long d'une trajectoire enregistrée (reward.pkl), via
la RewardFunction de tmrl. C'est l'approche qui fait apprendre vite — signal dense
+ détection d'échec automatique (voiture qui ne progresse plus / trop loin de la
ligne). Le LIDAR sert uniquement d'OBSERVATION (pas de pénalité bricolée).

Observation : 4 frames empilées de [scalaires + 19 LIDAR] → le réseau perçoit le
mouvement (vitesse, rotation), pas juste une photo figée.

Pré-requis : TM2020 ouvert (PAS minimisé), TMRL_GrabData.op chargé, une course
lancée, et un reward.pkl correspondant à la map (celui de tmrl-train marche déjà).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time
from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import vgamepad as vg

from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient, Lidar
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.compute_reward import RewardFunction

import config

log = logging.getLogger(__name__)

_SPEED, _DIST = 0, 1
_X, _Y, _Z = 2, 3, 4
_STEER, _GAS, _BRAKE = 5, 6, 7
_FINISHED, _GEAR, _RPM = 8, 9, 10

N_LIDAR   = 19
IMG_W     = 160
IMG_H     = 120
LIDAR_MAX = 200.0
SPEED_MAX = 110.0

SINGLE_OBS = 9 + N_LIDAR              # une frame : 9 scalaires + 19 LIDAR = 28
OBS_SIZE   = SINGLE_OBS * config.FRAME_STACK


class TrackmaniaEnv(gym.Env):
    """
    Observation : FRAME_STACK frames concaténées, chacune =
        [speed, posx, posy, posz, steer, gas, brake, gear, rpm] + lidar[19]

    Action (3 floats en [-1, 1]) :
        action[0]  direction (stick gauche X)
        action[1]  gaz       → (action[1]+1)/2 ∈ [0,1], défaut 0.5 (biaisé « on »
                              pour éviter la voiture immobile en exploration)
        action[2]  frein     → clip(action[2], 0, 1)

    L'agent peut enfin LEVER LE PIED / freiner avant les virages au lieu de foncer
    plein gaz dedans → indispensable pour ne plus clipper les murs en courbe.
    """

    metadata = {"render_modes": []}

    def __init__(self, port: int = config.OPENPLANET_PORT) -> None:
        super().__init__()

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
        )

        log.info("Connexion au plugin TMRL_GrabData.op (port %d) …", port)
        self.client = TM2020OpenPlanetClient(port=port)

        log.info("Recherche de la fenêtre 'Trackmania' pour le LIDAR …")
        self.window = WindowInterface("Trackmania")
        self._lidar: Lidar | None = None

        log.info("Chargement de la trajectoire de récompense : %s", config.REWARD_PKL)
        self.reward_fn = RewardFunction(
            reward_data_path=config.REWARD_PKL,
            nb_obs_forward=config.CHECK_FORWARD,
            nb_obs_backward=config.CHECK_BACKWARD,
            nb_zero_rew_before_failure=config.FAILURE_COUNTDOWN,
            min_nb_steps_before_failure=config.MIN_STEPS,
            max_dist_from_traj=config.MAX_STRAY,
        )

        log.info("Initialisation du gamepad virtuel Xbox 360 …")
        self.gamepad = vg.VX360Gamepad()
        self._release_all()

        self._stack: deque = deque(maxlen=config.FRAME_STACK)
        self._ep_start: float = 0.0
        self._next_deadline: float = 0.0
        self._step_count: int = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_all()
        self._press_abandon()
        time.sleep(config.RESET_WAIT_S)
        self.reward_fn.reset()

        data  = self.client.retrieve_data()
        lidar = self._get_lidar()
        single = self._single_obs(data, lidar)

        self._stack.clear()
        for _ in range(config.FRAME_STACK):
            self._stack.append(single)

        now = time.monotonic()
        self._ep_start      = now
        self._next_deadline = now + config.CONTROL_TIMESTEP
        return self._stacked_obs(), {}

    def step(self, action: np.ndarray):
        self._apply_action(action)
        data  = self.client.retrieve_data()
        lidar = self._get_lidar()
        self._maintain_frequency()

        # Récompense de trajectoire (tmrl) + détection d'échec
        pos = np.array([data[_X], data[_Y], data[_Z]])
        traj_rew, failed = self.reward_fn.compute_reward(pos)

        finished  = bool(data[_FINISHED] > 0.5)
        timed_out = (time.monotonic() - self._ep_start) > config.EPISODE_TIMEOUT
        speed_ms  = float(data[_SPEED])

        reward = config.K_TRAJ * float(traj_rew) + config.K_SPEED * (speed_ms / SPEED_MAX)
        if finished:
            reward += config.FINISH_BONUS

        terminated = finished
        truncated  = bool(failed or timed_out)

        self._stack.append(self._single_obs(data, lidar))

        info = {
            "speed_kmh": speed_ms * 3.6,
            "traj_reward": float(traj_rew),
            "finished": finished,
            "stuck": bool(failed),
        }
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True   # bootstrap → pas de peur du crash

        self._step_count += 1
        if self._step_count % 100 == 0:
            log.info(
                "step %d | %3.0f km/h | traj_rew=%+.2f | idx=%d/%d",
                self._step_count, info["speed_kmh"], float(traj_rew),
                self.reward_fn.cur_idx, self.reward_fn.datalen,
            )

        return self._stacked_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        self._release_all()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._stack)).astype(np.float32)

    def _maintain_frequency(self) -> None:
        sleep_left = self._next_deadline - time.monotonic()
        if sleep_left > 0:
            time.sleep(sleep_left)
        self._next_deadline += config.CONTROL_TIMESTEP
        if self._next_deadline < time.monotonic():
            self._next_deadline = time.monotonic() + config.CONTROL_TIMESTEP

    def _get_lidar(self) -> np.ndarray:
        try:
            img = self.window.screenshot()[:, :, :3]
            img = cv2.resize(img, (IMG_W, IMG_H))
            if self._lidar is None:
                self._lidar = Lidar(img)
            raw = self._lidar.lidar_20(img)
            return np.clip(raw / LIDAR_MAX, 0.0, 1.0).astype(np.float32)
        except Exception as exc:
            log.debug("LIDAR échoué : %s", exc)
            return np.full(N_LIDAR, 0.5, dtype=np.float32)

    def _single_obs(self, data, lidar: np.ndarray) -> np.ndarray:
        core = np.array([
            data[_SPEED] / SPEED_MAX,
            data[_X] / 1000.0,
            data[_Y] / 1000.0,
            data[_Z] / 1000.0,
            data[_STEER],
            data[_GAS],
            data[_BRAKE],
            data[_GEAR] / 6.0,
            data[_RPM] / 10000.0,
        ], dtype=np.float32)
        return np.concatenate([core, lidar])

    def _apply_action(self, action: np.ndarray) -> None:
        steer = float(np.clip(action[0], -1.0, 1.0))
        gas   = float((np.clip(action[1], -1.0, 1.0) + 1.0) / 2.0)  # 0..1, défaut 0.5
        brake = float(np.clip(action[2], 0.0, 1.0))
        self.gamepad.right_trigger_float(value_float=gas)
        self.gamepad.left_trigger_float(value_float=brake)
        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.gamepad.update()

    def _release_all(self) -> None:
        self.gamepad.reset()
        self.gamepad.update()

    def _press_abandon(self) -> None:
        """Envoie SUPPR à TM2020 (= redépart depuis le début)."""
        import win32gui
        import ctypes

        hwnd = win32gui.FindWindow(None, "Trackmania")
        if not hwnd:
            log.warning("Fenêtre 'Trackmania' introuvable — reset impossible.")
            return
        ctypes.windll.user32.SetForegroundWindow(hwnd)
        time.sleep(0.15)
        from tmrl.custom.tm.utils.control_keyboard import PressKey, ReleaseKey
        PressKey(0xD3)
        time.sleep(0.05)
        ReleaseKey(0xD3)
        time.sleep(0.1)
