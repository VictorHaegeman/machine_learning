"""
Enregistre une trajectoire de récompense pour une map TM2020 personnalisée.

À utiliser UNIQUEMENT pour une autre map que tmrl-train (qui a déjà son reward.pkl).

Étapes :
  1. Lance TM2020 sur ta map, mets-toi sur la ligne de départ.
  2. Lance :  python record_reward.py
  3. Conduis un tour complet (pas besoin d'être rapide, juste suivre la piste).
  4. À l'arrivée l'enregistrement s'arrête seul (ou Ctrl+C).
  5. Le fichier reward.pkl est écrit dans le dossier du projet.
  6. Mets dans config.py :  REWARD_PKL = "reward.pkl"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pickle
import time

import numpy as np

from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

OUT = "reward.pkl"
MIN_SPACING = 0.1   # mètres entre deux points enregistrés (≈ format tmrl)


def main() -> None:
    client = TM2020OpenPlanetClient(port=9000)
    print("Connecté au plugin. Conduis ton tour ! (Ctrl+C pour arrêter)\n")

    points: list[list[float]] = []
    last = None
    try:
        while True:
            data = client.retrieve_data()
            pos = np.array([data[2], data[3], data[4]])
            if last is None or np.linalg.norm(pos - last) >= MIN_SPACING:
                points.append([float(pos[0]), float(pos[1]), float(pos[2])])
                last = pos
                if len(points) % 100 == 0:
                    print(f"\r{len(points)} points enregistrés…", end="", flush=True)
            if data[8] > 0.5 and len(points) > 50:   # arrivée franchie
                print("\nArrivée détectée — fin de l'enregistrement.")
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nArrêt manuel.")

    if len(points) < 50:
        print(f"Trop peu de points ({len(points)}). Recommence en conduisant un vrai tour.")
        return

    arr = np.array(points, dtype=np.float64)
    with open(OUT, "wb") as f:
        pickle.dump(arr, f)

    length = np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()
    print(f"\n✓ {OUT} écrit : {len(arr)} points, ~{length:.0f} m de trajectoire.")
    print('  → Dans config.py, mets :  REWARD_PKL = "reward.pkl"')


if __name__ == "__main__":
    main()
