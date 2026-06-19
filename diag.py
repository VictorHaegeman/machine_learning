"""
Diagnostic de perception — capture ce que le bot "voit" et teste le LIDAR.

Lancer TM2020 sur une course (voiture visible), puis :
    python diag.py

Génère :
    diag_raw.png    capture brute de la fenêtre Trackmania
    diag_small.png  image redimensionnée utilisée par le LIDAR (160x120)
    diag_lidar.png  même image + rayons LIDAR dessinés
et affiche les statistiques de couleur + les 19 distances LIDAR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar, armin

IMG_W, IMG_H = 160, 120

def main() -> None:
    win = WindowInterface("Trackmania")
    raw = win.screenshot()[:, :, :3]            # BGR
    cv2.imwrite("diag_raw.png", raw)

    small = cv2.resize(raw, (IMG_W, IMG_H))
    cv2.imwrite("diag_small.png", small)

    print(f"Capture brute : {raw.shape}  (h, w, c)")
    print(f"Couleur moyenne BGR (brute)  : {raw.reshape(-1,3).mean(0).round(1)}")
    print(f"Couleur moyenne BGR (petite) : {small.reshape(-1,3).mean(0).round(1)}")

    # % de pixels considérés "noirs" (< 55 sur les 3 canaux) par le LIDAR
    dark = np.all(small < 55, axis=2).mean() * 100
    print(f"Pixels 'noirs' (< 55) détectés comme murs : {dark:.0f}%")

    # LIDAR
    lid = Lidar(small)
    rp = lid.road_point
    print(f"Point de départ des rayons (ligne, col) : {rp}")
    dists = lid.lidar_20(small)
    print(f"19 distances LIDAR : {np.round(dists,1)}")
    print(f"  min={dists.min():.1f}  max={dists.max():.1f}  moyenne={dists.mean():.1f}")

    # Dessine les rayons pour visualisation
    ann = small.copy()
    cv2.circle(ann, (rp[1], rp[0]), 2, (0, 0, 255), -1)
    for ax_x, ax_y in zip(lid.list_axis_x, lid.list_axis_y):
        if len(ax_x) == 0:
            continue
        idx = armin(np.all(small[ax_x, ax_y] < lid.black_threshold, axis=1))
        cv2.line(ann, (rp[1], rp[0]), (int(ax_y[idx]), int(ax_x[idx])), (255, 0, 0), 1)
    cv2.imwrite("diag_lidar.png", ann)

    print("\nImages écrites : diag_raw.png, diag_small.png, diag_lidar.png")
    if dists.max() < 1.0:
        print(">>> Tous les rayons à ~0 : l'image est trop sombre ou la capture a échoué.")
    elif dark > 40:
        print(">>> Beaucoup de noir : le décor sombre trompe le LIDAR → essayer tmrl-train.")
    else:
        print(">>> Le LIDAR détecte des distances variées : perception OK.")


if __name__ == "__main__":
    main()
