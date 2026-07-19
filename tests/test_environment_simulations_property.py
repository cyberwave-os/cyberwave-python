"""cw.environments.simulations exposure + managers package exports."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_environment_manager_exposes_simulations() -> None:
    from cyberwave.managers.simulations import SimulationManager
    from cyberwave.resources import EnvironmentManager

    mgr = EnvironmentManager(MagicMock())
    sims = mgr.simulations
    assert isinstance(sims, SimulationManager)
    assert sims.api is mgr.api
    # cached (same instance on repeated access)
    assert mgr.simulations is sims


def test_managers_package_reexports_simulation_symbols() -> None:
    from cyberwave.managers import Simulation, SimulationManager

    assert SimulationManager is not None
    assert Simulation is not None
