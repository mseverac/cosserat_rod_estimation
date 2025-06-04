from scipy.optimize import minimize
import numpy as np
import time   # <-- pour mesurer le temps

def minimize_with_early_stop(
    fun,
    x0,
    threshold=1e-3,
    max_time=None,          # ← nouveau paramètre (en secondes, None ⇢ pas de limite)
    method='L-BFGS-B',
    bounds=None,
    options=None,
):
    """
    Minimisation avec arrêt anticipé :
      • si f(x) passe sous `threshold`
      • OU si la durée dépasse `max_time` (en secondes)

    Renvoie un objet possédant .x, .fun, .success, .message
    """

    best_solution = {'x': None, 'fun': np.inf}

    class EarlyStoppingException(Exception):
        def __init__(self, reason):
            self.reason = reason

    start_time = time.monotonic()               # chrono au lancement

    def early_stop_callback(xk):
        nonlocal start_time
        fx = fun(xk)
        # mémoriser le meilleur point rencontré
        if fx < best_solution['fun']:
            best_solution['x'] = xk.copy()
            best_solution['fun'] = fx

        # condition 1 : objectif sous le seuil
        if fx < threshold:
            raise EarlyStoppingException("threshold reached")

        # condition 2 : temps écoulé
        if (max_time is not None) and (time.monotonic() - start_time > max_time):
            raise EarlyStoppingException("time limit reached")

    try:
        result = minimize(
            fun=fun,
            x0=x0,
            method=method,
            bounds=bounds,
            options=options,
            callback=early_stop_callback,
        )
    except EarlyStoppingException as e:
        reason = e.reason
        class ResultMock:
            def __init__(self, x, fun, reason):
                self.x = x
                self.fun = fun
                self.success = False
                self.message = f"Stopped early: {reason}."
        result = ResultMock(best_solution['x'], best_solution['fun'], reason)

    return result
