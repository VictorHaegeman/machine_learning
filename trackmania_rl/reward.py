def compute_reward(
    speed_kmh: float,
    new_checkpoint: bool,
    finished: bool,
    stuck: bool,
) -> float:
    """
    Returns the scalar reward for one environment step.

    Design rationale
    ----------------
    - speed bonus   : tiny per-step push to go fast without dominating the signal
    - checkpoint    : main learning signal — reach the next checkpoint
    - finish        : large bonus when the full lap is completed
    - time penalty  : constant −0.01 discourages dawdling
    - stuck penalty : punishes episodes where the car makes no progress
    """
    reward = 0.0

    # Small speed bonus (max ~0.03/step at 300 km/h)
    reward += speed_kmh * 0.0001

    if new_checkpoint:
        reward += 10.0

    if finished:
        reward += 100.0

    # Constant time penalty
    reward -= 0.01

    if stuck:
        reward -= 1.0

    return reward
