"""Stores custom logging/main loops."""

from logging import Logger

import dedalus
import dedalus.public as d3


def track_vorticity(
    logger: Logger,
    flow: d3.GlobalFlowProperty,
    solver: dedalus.core.solvers.InitialValueSolver,
<<<<<<< HEAD
    CFL: d3.CFL,  # noqa: N803 (Allows argument to be capitalised)
=======
    cfl: d3.CFL,
>>>>>>> main
) -> None:
    """
    Create main loop that tracks and logs the maximum superfluid vorticity.

    Should be called as an alternative to solver.evolve.

    :param logger: Logger used by the script.
    :param flow: dedalus flow object. Must track the maximum vorticity as vorticity_mag.
    :param solver: The IVP solver defined by the script.
<<<<<<< HEAD
    :param CFL: The CFL condition used by the script.
=======
    :param cfl: The CFL condition used by the script.
>>>>>>> main
    """
    try:
        logger.info("Starting main loop")
        while solver.proceed:
<<<<<<< HEAD
            timestep = CFL.compute_timestep()
=======
            timestep = cfl.compute_timestep()
>>>>>>> main
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
<<<<<<< HEAD


def track_reynolds_n(
    logger: Logger,
    flow: d3.GlobalFlowProperty,
    solver: dedalus.core.solvers.InitialValueSolver,
    CFL: d3.CFL,  # noqa: N803 (Allows argument to be capitalised)
) -> None:
    """
    Create main loop that tracks and logs the maximum reynolds number.

    Should be called as an alternative to solver.evolve.

    :param logger: Logger used by the script.
    :param flow: dedalus flow object. Must track the normal fluid
    reynlods number as Re_n.
    :param solver: The IVP solver defined by the script.
    :param CFL: The CFL condition used by the script.
    """
    try:
        logger.info("Starting main loop")
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration - 1) % 10 == 0:
                re = flow.max("Re_n")
                logger.info(
                    "Iteration=%i, Time=%e, dt=%e, max(Re)=%f"
                    % (solver.iteration, solver.sim_time, timestep, re)
                )
    except:
        logger.exception("Exception raised, triggering end of main loop.")
        raise
    finally:
        solver.log_stats()
=======
>>>>>>> main
