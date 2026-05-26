"""Stores custom logging/main loops."""

from logging import Logger

import dedalus
import dedalus.public as d3


def track_vorticity(
    logger: Logger,
    flow: d3.GlobalFlowProperty,
    solver: dedalus.core.solvers.InitialValueSolver,
    cfl: d3.CFL,
) -> None:
    """
    Create main loop that tracks and logs the maximum superfluid vorticity.

    Should be called as an alternative to solver.evolve.

    :param logger: Logger used by the script.
    :param flow: dedalus flow object. Must track the maximum vorticity as vorticity_mag.
    :param solver: The IVP solver defined by the script.
    :param cfl: The CFL condition used by the script.
    """
    try:
        logger.info("Starting main loop")
        while solver.proceed:
            timestep = cfl.compute_timestep()
            solver.step(timestep)
            if (solver.iteration - 1) % 10 == 0:
                max_omega = flow.max("vorticity_mag")
                logger.info(
                    "Iteration=%i, Time=%e, dt=%e, max(omega_s)=%f"
                    % (solver.iteration, solver.sim_time, timestep, max_omega)
                )
    except:
        logger.exception("Exception raised, triggering end of main loop.")
        raise
    finally:
        solver.log_stats()
