import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions


# swapping PAUSE for STOP
HabitatSimActions.extend_action_space("PAUSE")
temp = HabitatSimActions.STOP
HabitatSimActions._known_actions["STOP"] = HabitatSimActions.PAUSE
HabitatSimActions._known_actions["PAUSE"] = temp


@registry.register_action_space_configuration(name="rir-rendering")
class RIRRenderingActionSpaceConfiguration(ActionSpaceConfiguration):
    def get(self):        
        return {
            HabitatSimActions.PAUSE: habitat_sim.ActionSpec("pause"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }
