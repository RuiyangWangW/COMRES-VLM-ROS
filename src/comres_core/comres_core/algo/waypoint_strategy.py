"""
Waypoint assignment strategy interface: the simulator consults its strategy every step
(assign_waypoints), and every strategy decides itself when to coordinate (its T_coord / T_H).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class WaypointStrategy(ABC):
    """
    Base class for waypoint assignment strategies.
    """

    # True when every waypoint the strategy hands out is a frontier representative, so a
    # robot may count it reached once its frontier has vanished (Robot.vanish_reached).
    frontier_waypoints = False

    @abstractmethod
    def assign_waypoints(
        self,
        robots: List,
        step_count: int,
        **kwargs
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Assign waypoints to robots.

        Args:
            robots: List of Robot objects
            step_count: Current simulation step
            **kwargs: Additional strategy-specific parameters
                     (unused by the current strategies, which are self-contained)

        Returns:
            Dict[robot_index] -> list of (row, col) waypoints ({} = nothing new)
        """
        pass

    def reset(self):
        """
        Reset strategy state (called when loading a new scene).
        """
