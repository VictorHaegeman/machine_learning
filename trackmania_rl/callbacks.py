import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrackmaniaCallback(BaseCallback):
    """
    Métriques Trackmania loguées vers TensorBoard toutes les 10 fins d'épisode :
      - trackmania/finish_rate        proportion d'épisodes terminés (arrivée)
      - trackmania/mean_speed_kmh     vitesse moyenne (1000 derniers steps)
      - trackmania/stuck_rate         proportion d'épisodes finis en échec
      - trackmania/ep_traj_reward     récompense de trajectoire cumulée / épisode
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._ep_finished: list[bool] = []
        self._ep_stuck: list[bool] = []
        self._step_speeds: list[float] = []
        self._ep_traj: list[float] = []
        self._cur_traj: float = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            self._step_speeds.append(float(info.get("speed_kmh", 0.0)))
            self._cur_traj += float(info.get("traj_reward", 0.0))
            if done:
                self._ep_finished.append(bool(info.get("finished", False)))
                self._ep_stuck.append(bool(info.get("stuck", False)))
                self._ep_traj.append(self._cur_traj)
                self._cur_traj = 0.0

        n = len(self._ep_finished)
        if n > 0 and n % 10 == 0:
            self.logger.record("trackmania/finish_rate", float(np.mean(self._ep_finished[-10:])))
            self.logger.record("trackmania/stuck_rate", float(np.mean(self._ep_stuck[-10:])))
            self.logger.record("trackmania/ep_traj_reward", float(np.mean(self._ep_traj[-10:])))
            self.logger.record(
                "trackmania/mean_speed_kmh",
                float(np.mean(self._step_speeds[-1000:])) if self._step_speeds else 0.0,
            )

        return True
