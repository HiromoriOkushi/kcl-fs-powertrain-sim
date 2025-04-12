"""
Shift strategy module for Formula Student powertrain simulation.

Defines different strategies for automatic gear shifting based on performance
goals like maximum acceleration, efficiency, or endurance.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import os
import yaml

# Assuming TorqueCurve might be used for advanced strategies
try:
    from ..engine.torque_curve import TorqueCurve
except ImportError:
    class TorqueCurve: pass # Placeholder
    TorqueCurve = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ShiftStrategy")

class StrategyType(Enum):
    """Types of shift strategies."""
    MAX_ACCELERATION = auto()
    MAX_EFFICIENCY = auto()
    ENDURANCE = auto()
    ACCELERATION = auto() # Specific for accel event
    SKIDPAD = auto()
    AUTOCROSS = auto()
    MANUAL = auto() # Represents manual control
    CUSTOM = auto()

class ShiftCondition(Enum):
    """Conditions triggering a shift."""
    RPM_THRESHOLD = auto()
    SPEED_THRESHOLD = auto()
    LOAD_THRESHOLD = auto()
    # Advanced conditions (require more data/models)
    TORQUE_CROSSOVER = auto() # Torque in next gear > torque in current gear
    POWER_CROSSOVER = auto() # Power in next gear > power in current gear
    OPTIMAL_ACCELERATION = auto() # Based on calculated acceleration in gears
    TIME_SINCE_START = auto() # For timed events like launch
    CUSTOM = auto()

class ShiftPoint:
    """Defines a specific condition for initiating a gear shift."""
    def __init__(self, from_gear: int, to_gear: int,
                 condition: ShiftCondition, threshold: float,
                 priority: int = 1, description: Optional[str] = None):
        """
        Args:
            from_gear: The gear shifting from.
            to_gear: The gear shifting to.
            condition: The ShiftCondition enum type.
            threshold: The value associated with the condition (e.g., RPM, speed).
            priority: Higher value means higher priority (checked first).
            description: Optional description.
        """
        self.from_gear = from_gear
        self.to_gear = to_gear
        self.condition = condition
        self.threshold = threshold
        self.priority = priority
        self.description = description if description else \
                           f"{condition.name} {'<' if to_gear < from_gear else '>'} {threshold:.1f}"

    def __str__(self) -> str:
        direction = "Upshift" if self.to_gear > self.from_gear else "Downshift"
        return f"{direction} {self.from_gear}->{self.to_gear} when {self.description} (Prio: {self.priority})"

    def __lt__(self, other: 'ShiftPoint') -> bool:
        # For sorting by priority (higher priority first)
        return self.priority > other.priority


class ShiftStrategy:
    """Base class for all shift strategies."""
    def __init__(self, strategy_type: StrategyType, name: str = ""):
        self.strategy_type = strategy_type
        self.name = name or strategy_type.name.replace('_', ' ').title()
        # Store shift points: {from_gear: [ShiftPoint1, ShiftPoint2, ...]} sorted by priority
        self.shift_points: Dict[int, List[ShiftPoint]] = {}
        self.shift_log: List[Dict] = []
        logger.info(f"Initialized Shift Strategy: {self.name}")

    def add_shift_point(self, shift_point: ShiftPoint):
        """Add a shift point rule to the strategy."""
        from_gear = shift_point.from_gear
        if from_gear not in self.shift_points:
            self.shift_points[from_gear] = []
        self.shift_points[from_gear].append(shift_point)
        self.shift_points[from_gear].sort() # Sort by priority (desc)
        logger.debug(f"Added shift point to {self.name}: {shift_point}")

    def _check_condition(self, point: ShiftPoint, state: Dict) -> bool:
        """Check if a specific shift point condition is met."""
        condition = point.condition
        threshold = point.threshold
        direction = 'up' if point.to_gear > point.from_gear else 'down'

        # Required state keys (add more as needed by conditions)
        current_rpm = state.get('engine_rpm')
        current_speed = state.get('vehicle_speed')
        current_load = state.get('engine_load') # Assume 0-1 load factor

        if condition == ShiftCondition.RPM_THRESHOLD:
            if current_rpm is None: return False
            return current_rpm >= threshold if direction == 'up' else current_rpm <= threshold
        elif condition == ShiftCondition.SPEED_THRESHOLD:
            if current_speed is None: return False
            return current_speed >= threshold if direction == 'up' else current_speed <= threshold
        elif condition == ShiftCondition.LOAD_THRESHOLD:
            if current_load is None: return False
            # Example: Upshift if load is low, downshift if load is high (can be customized)
            return current_load <= threshold if direction == 'up' else current_load >= threshold
        # --- Add checks for TORQUE/POWER/ACCELERATION crossover ---
        # These require predicting performance in the target gear, needing engine/drivetrain info in 'state'
        elif condition in [ShiftCondition.TORQUE_CROSSOVER, ShiftCondition.POWER_CROSSOVER, ShiftCondition.OPTIMAL_ACCELERATION]:
             logger.warning(f"Condition {condition.name} requires advanced prediction - not fully implemented in base check.")
             # Placeholder logic: Shift near redline for these for now
             if direction == 'up' and current_rpm is not None:
                 engine_redline = state.get('engine_redline_rpm', 14000)
                 return current_rpm >= engine_redline * 0.95
             return False # Don't trigger downshifts based on this placeholder
        elif condition == ShiftCondition.TIME_SINCE_START:
             if 'elapsed_time_s' not in state: return False
             return state['elapsed_time_s'] >= threshold
        elif condition == ShiftCondition.CUSTOM:
             if 'custom_eval_func' not in state or not callable(state['custom_eval_func']): return False
             return state['custom_eval_func'](point, state) # Call custom function

        return False # Default false for unimplemented conditions

    def evaluate_shift(self, current_gear: int, state: Dict) -> Optional[int]:
        """
        Determine the target gear based on the current state and strategy rules.

        Args:
            current_gear: The current gear (0=N).
            state: Dictionary containing current vehicle state (rpm, speed, load, etc.).

        Returns:
            Target gear number (int) or None if no shift is recommended.
        """
        if current_gear not in self.shift_points:
            return None # No rules defined for shifting *from* this gear

        # Check rules for the current gear, ordered by priority
        for point in self.shift_points[current_gear]:
            if self._check_condition(point, state):
                # Check if target gear is valid
                num_gears = state.get('num_gears', 6) # Get from state or assume 6
                if 0 <= point.target_gear <= num_gears:
                    logger.debug(f"{self.name}: Recommending shift {current_gear}->{point.target_gear} due to {point.description}")
                    return point.target_gear
                else:
                    logger.warning(f"Invalid target gear {point.target_gear} defined in strategy {self.name}")

        return None # No shift condition met

    def record_shift(self, from_gear: int, to_gear: int, state: Dict, timestamp: float):
        """Log a shift event."""
        log_entry = {
            'timestamp': timestamp,
            'from_gear': from_gear,
            'to_gear': to_gear,
            'engine_rpm': state.get('engine_rpm'),
            'vehicle_speed': state.get('vehicle_speed'),
            'engine_load': state.get('engine_load'),
            'throttle': state.get('throttle_position')
        }
        self.shift_log.append(log_entry)

    def analyze_performance(self) -> Dict:
        """Analyze logged shift data."""
        if not self.shift_log: return {'total_shifts': 0}
        df = pd.DataFrame(self.shift_log)
        upshifts = df[df['to_gear'] > df['from_gear']]
        downshifts = df[df['to_gear'] < df['from_gear']]
        return {
            'total_shifts': len(df),
            'num_upshifts': len(upshifts),
            'num_downshifts': len(downshifts),
            'avg_upshift_rpm': upshifts['engine_rpm'].mean() if not upshifts.empty else None,
            'avg_downshift_rpm': downshifts['engine_rpm'].mean() if not downshifts.empty else None,
            'shift_frequency_hz': len(df) / (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) if len(df) > 1 else 0
        }

    def plot_shift_points(self, state_data: pd.DataFrame, save_path: Optional[str] = None):
         """Plot shift points over a time series of state data."""
         from ..utils.plotting import save_plot # Local import

         if not self.shift_log:
             logger.warning("No shift history to plot for strategy.")
             return

         fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

         time = state_data['time']

         # Plot RPM and Shifts
         axes[0].plot(time, state_data['engine_rpm'], label='Engine RPM', color='blue')
         axes[0].set_ylabel('Engine RPM')
         axes[0].grid(True, alpha=0.5)
         shift_times = [s['timestamp'] for s in self.shift_log]
         shift_rpms = [s['engine_rpm'] for s in self.shift_log]
         shift_labels = [f"{s['from_gear']}->{s['to_gear']}" for s in self.shift_log]
         for t, rpm, lbl in zip(shift_times, shift_rpms, shift_labels):
              axes[0].scatter([t], [rpm], color='red', marker='o', s=50, zorder=5)
              axes[0].text(t, rpm + 200, lbl, color='red', ha='center', fontsize=8)
         axes[0].legend()

         # Plot Speed and Gear
         ax_speed = axes[1]
         ax_gear = ax_speed.twinx()
         ln1 = ax_speed.plot(time, state_data['vehicle_speed'] * 3.6, label='Speed (km/h)', color='green') # kph
         ax_speed.set_ylabel('Speed (km/h)', color='green')
         ax_speed.tick_params(axis='y', labelcolor='green')
         ax_speed.grid(True, alpha=0.5)

         ln2 = ax_gear.step(time, state_data['gear'], label='Gear', color='orange', where='post')
         ax_gear.set_ylabel('Gear', color='orange')
         ax_gear.tick_params(axis='y', labelcolor='orange')
         ax_gear.yaxis.set_major_locator(MaxNLocator(integer=True))
         ax_gear.set_ylim(0.5, max(state_data['gear']) + 0.5)

         lns = ln1 + [ln2] # Combine lines for legend
         labs = [l.get_label() for l in lns]
         ax_speed.legend(lns, labs, loc='center left')


         # Plot Throttle and Load
         axes[2].plot(time, state_data['throttle_position'], label='Throttle', color='purple')
         axes[2].plot(time, state_data['engine_load'], label='Load', color='brown', linestyle='--')
         axes[2].set_xlabel('Time (s)')
         axes[2].set_ylabel('Input (0-1)')
         axes[2].set_ylim(-0.05, 1.05)
         axes[2].grid(True, alpha=0.5)
         axes[2].legend()

         fig.suptitle(f'Shift Strategy Analysis: {self.name}')
         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
         if save_path: save_plot(fig, save_path)
         plt.show()
         plt.close(fig)


# --- Specific Strategy Implementations ---

class MaxAccelerationStrategy(ShiftStrategy):
    """Shifts near redline to maximize power output."""
    def __init__(self, engine_redline_rpm: float, num_gears: int):
        super().__init__(StrategyType.MAX_ACCELERATION)
        upshift_rpm = engine_redline_rpm * 0.97 # Shift very close to redline
        downshift_rpm = engine_redline_rpm * 0.60 # Downshift earlier to get back into power band

        for i in range(1, num_gears):
            self.add_shift_point(ShiftPoint(i, i + 1, ShiftCondition.RPM_THRESHOLD, upshift_rpm))
            self.add_shift_point(ShiftPoint(i + 1, i, ShiftCondition.RPM_THRESHOLD, downshift_rpm))


class MaxEfficiencyStrategy(ShiftStrategy):
    """Shifts at lower RPMs, typically just after peak torque, to save fuel."""
    def __init__(self, engine_peak_torque_rpm: float, num_gears: int, min_rpm: float = 2500):
        super().__init__(StrategyType.MAX_EFFICIENCY)
        upshift_rpm = engine_peak_torque_rpm * 1.10 # Shift slightly after peak torque
        downshift_rpm = min_rpm # Downshift to avoid lugging

        for i in range(1, num_gears):
            self.add_shift_point(ShiftPoint(i, i + 1, ShiftCondition.RPM_THRESHOLD, upshift_rpm))
            # Need gear ratios to calculate accurate downshift RPM to land near min_rpm
            # Simplified: use a fixed low RPM threshold to trigger downshift
            self.add_shift_point(ShiftPoint(i + 1, i, ShiftCondition.RPM_THRESHOLD, downshift_rpm * 1.1)) # Downshift if RPM drops low


class EnduranceStrategy(ShiftStrategy):
    """Balanced strategy for endurance: good performance, efficiency, and less wear."""
    def __init__(self, engine_peak_power_rpm: float, engine_peak_torque_rpm: float, num_gears: int, min_rpm: float = 3500):
        super().__init__(StrategyType.ENDURANCE)
        # Shift somewhere between peak torque and peak power
        upshift_rpm = (engine_peak_power_rpm * 0.85 + engine_peak_torque_rpm * 1.15) / 2
        downshift_rpm = min_rpm * 1.2 # Downshift to stay comfortably above min RPM

        for i in range(1, num_gears):
            self.add_shift_point(ShiftPoint(i, i + 1, ShiftCondition.RPM_THRESHOLD, upshift_rpm))
            self.add_shift_point(ShiftPoint(i + 1, i, ShiftCondition.RPM_THRESHOLD, downshift_rpm))
            # Could add load-based criteria with lower priority


class AccelerationEventStrategy(ShiftStrategy):
    """Strategy specifically for the 75m acceleration event."""
    def __init__(self, engine_redline_rpm: float, num_gears: int):
        super().__init__(StrategyType.ACCELERATION)
        # Max acceleration strategy - shift just before limiter
        upshift_rpm = engine_redline_rpm * 0.98

        for i in range(1, num_gears):
             self.add_shift_point(ShiftPoint(i, i + 1, ShiftCondition.RPM_THRESHOLD, upshift_rpm))
        # No downshifts needed for accel event typically

        # Launch control parameters (can be added/configured separately)
        self.launch_rpm: Optional[float] = None
        self.launch_slip_target: Optional[float] = None

    def configure_launch_control(self, launch_rpm: float, slip_target: float):
        self.launch_rpm = launch_rpm
        self.launch_slip_target = slip_target

    def get_launch_params(self) -> Optional[Dict]:
        if self.launch_rpm and self.launch_slip_target:
            return {'launch_rpm': self.launch_rpm, 'slip_target': self.launch_slip_target}
        return None


class SkidpadStrategy(ShiftStrategy):
    """Strategy for Skidpad: Hold a single optimal gear."""
    def __init__(self, target_gear: int, num_gears: int, high_rpm_thresh: float = 12000, low_rpm_thresh: float = 4000):
        super().__init__(StrategyType.SKIDPAD)
        self.target_gear = target_gear
        # Add rules to shift *into* the target gear and *stay* there
        # Upshift to target gear
        for i in range(1, target_gear):
             self.add_shift_point(ShiftPoint(i, target_gear, ShiftCondition.RPM_THRESHOLD, high_rpm_thresh))
        # Downshift to target gear
        for i in range(target_gear + 1, num_gears + 1):
             self.add_shift_point(ShiftPoint(i, target_gear, ShiftCondition.RPM_THRESHOLD, low_rpm_thresh))


class AutocrossStrategy(ShiftStrategy):
    """Dynamic strategy for autocross, similar to Endurance but potentially more aggressive."""
    def __init__(self, engine_peak_power_rpm: float, engine_peak_torque_rpm: float, num_gears: int, min_rpm: float = 4000):
         super().__init__(StrategyType.AUTOCROSS)
         # Slightly more aggressive than endurance
         upshift_rpm = engine_peak_power_rpm * 0.92
         downshift_rpm = min_rpm * 1.15

         for i in range(1, num_gears):
             self.add_shift_point(ShiftPoint(i, i + 1, ShiftCondition.RPM_THRESHOLD, upshift_rpm))
             self.add_shift_point(ShiftPoint(i + 1, i, ShiftCondition.RPM_THRESHOLD, downshift_rpm))


class StrategyManager:
    """Manages multiple shift strategies and selects the active one."""
    def __init__(self, default_strategy: Optional[ShiftStrategy] = None):
        self.strategies: Dict[str, ShiftStrategy] = {}
        self.active_strategy: Optional[ShiftStrategy] = None
        if default_strategy:
            self.add_strategy(default_strategy)
            self.set_active_strategy(default_strategy.name)
        logger.info("Shift Strategy Manager initialized.")

    def add_strategy(self, strategy: ShiftStrategy):
        """Add a strategy."""
        if not isinstance(strategy, ShiftStrategy): raise TypeError("Input must be ShiftStrategy.")
        self.strategies[strategy.name] = strategy
        logger.info(f"Strategy '{strategy.name}' added.")
        if self.active_strategy is None: # Set first added as active if none is set
             self.set_active_strategy(strategy.name)

    def set_active_strategy(self, name: str) -> bool:
        """Set the active strategy by name."""
        if name in self.strategies:
            self.active_strategy = self.strategies[name]
            logger.info(f"Active strategy set to: {name}")
            return True
        else:
            logger.error(f"Strategy '{name}' not found.")
            return False

    def evaluate_shift(self, current_gear: int, state: Dict) -> Optional[int]:
        """Evaluate shift using the active strategy."""
        if self.active_strategy:
            return self.active_strategy.evaluate_shift(current_gear, state)
        else:
            logger.warning("No active strategy set, cannot evaluate shift.")
            return None

    def record_shift(self, from_gear: int, to_gear: int, state: Dict, timestamp: float):
         """Record shift in the active strategy."""
         if self.active_strategy:
              self.active_strategy.record_shift(from_gear, to_gear, state, timestamp)

    def get_active_strategy_name(self) -> Optional[str]:
        """Get the name of the currently active strategy."""
        return self.active_strategy.name if self.active_strategy else None

    def load_strategies_from_config(self, config_path: str):
        """Load multiple strategies defined in a YAML config file."""
        if not os.path.exists(config_path):
             logger.error(f"Strategy config file not found: {config_path}")
             return
        try:
            with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)

            engine_params = config.get('engine', {})
            max_rpm = engine_params.get('max_rpm', 14000)
            peak_power_rpm = engine_params.get('peak_power_rpm', 12500)
            peak_torque_rpm = engine_params.get('peak_torque_rpm', 10500)
            idle_rpm = engine_params.get('idle_rpm', 1300)
            num_gears = config.get('num_gears', 6) # Need num_gears or gear_ratios

            strategy_configs = config.get('strategies', {})
            for name, params in strategy_configs.items():
                 strategy_type_str = params.get('type', name).upper()
                 try:
                     strategy_type = StrategyType[strategy_type_str]
                     # Create strategy based on type - needs refinement based on constructor args
                     if strategy_type == StrategyType.MAX_ACCELERATION:
                          strat = MaxAccelerationStrategy(max_rpm, peak_power_rpm, num_gears) # Needs gear ratios ideally
                     elif strategy_type == StrategyType.MAX_EFFICIENCY:
                          strat = MaxEfficiencyStrategy(peak_torque_rpm, num_gears, idle_rpm + 1200)
                     elif strategy_type == StrategyType.ENDURANCE:
                          strat = EnduranceStrategy(max_rpm, peak_power_rpm, peak_torque_rpm, num_gears)
                     elif strategy_type == StrategyType.ACCELERATION:
                          strat = AccelerationEventStrategy(max_rpm, peak_power_rpm, num_gears) # Needs more args
                     # Add other types...
                     else:
                          strat = ShiftStrategy(strategy_type, name) # Generic base

                     # TODO: Add logic to parse and add custom shift points from config 'params'
                     # for point_def in params.get('shift_points', []):
                     #    sp = ShiftPoint(...)
                     #    strat.add_shift_point(sp)

                     self.add_strategy(strat)

                 except KeyError:
                      logger.error(f"Invalid strategy type '{strategy_type_str}' in config.")
                 except Exception as e:
                      logger.error(f"Error creating strategy '{name}': {e}")

            default_strategy = config.get('default_strategy')
            if default_strategy:
                 self.set_active_strategy(default_strategy)

        except Exception as e:
            logger.error(f"Error loading strategies from config {config_path}: {e}")


# Factory function
def create_formula_student_strategies(
    engine_max_rpm: float,
    engine_peak_power_rpm: float,
    engine_peak_torque_rpm: float,
    gear_ratios: List[float],
    num_gears: int, # Added num_gears explicitly
    idle_rpm: float = 1300,
    skidpad_gear: int = 2,
    autocross_min_rpm: float = 4000,
    efficiency_min_rpm: float = 2500,
    endurance_min_rpm: float = 3500
) -> StrategyManager:
    """Create standard set of FS strategies."""
    manager = StrategyManager()

    manager.add_strategy(MaxAccelerationStrategy(engine_redline_rpm=engine_max_rpm, num_gears=num_gears)) # Use redline
    manager.add_strategy(MaxEfficiencyStrategy(engine_peak_torque_rpm=engine_peak_torque_rpm, num_gears=num_gears, min_rpm=efficiency_min_rpm))
    manager.add_strategy(EnduranceStrategy(engine_peak_power_rpm=engine_peak_power_rpm, engine_peak_torque_rpm=engine_peak_torque_rpm, num_gears=num_gears, min_rpm=endurance_min_rpm))
    # AccelerationEventStrategy might need more args like wheel radius, mass if its logic uses them
    manager.add_strategy(AccelerationEventStrategy(engine_redline_rpm=engine_max_rpm, num_gears=num_gears)) # Use redline
    manager.add_strategy(SkidpadStrategy(target_gear=skidpad_gear, num_gears=num_gears))
    manager.add_strategy(AutocrossStrategy(engine_peak_power_rpm=engine_peak_power_rpm, engine_peak_torque_rpm=engine_peak_torque_rpm, num_gears=num_gears, min_rpm=autocross_min_rpm))

    manager.set_active_strategy("Endurance") # Default to endurance
    return manager


# Example Usage
if __name__ == "__main__":
    # Example parameters
    max_rpm=14000
    peak_power_rpm=12500
    peak_torque_rpm=10500
    gears=[2.750, 2.000, 1.667, 1.444, 1.304, 1.208]
    num_gears = len(gears)

    manager = create_formula_student_strategies(max_rpm, peak_power_rpm, peak_torque_rpm, gears, num_gears)

    print(f"Available strategies: {list(manager.strategies.keys())}")
    print(f"Active strategy: {manager.get_active_strategy_name()}")

    # Test evaluation
    state = {
        'engine_rpm': 13500,
        'vehicle_speed': 30, # m/s
        'engine_load': 0.9,
        'throttle_position': 1.0,
        'gear_ratios': gears,
        'num_gears': num_gears,
        'engine_redline_rpm': max_rpm
    }
    manager.set_active_strategy("Max Acceleration")
    target_gear = manager.evaluate_shift(current_gear=3, state=state)
    print(f"\nMax Accel eval at 13500 RPM in 3rd: Target Gear = {target_gear}") # Expect 4

    state['engine_rpm'] = 4000
    state['throttle_position'] = 0.7
    manager.set_active_strategy("Endurance")
    target_gear = manager.evaluate_shift(current_gear=5, state=state)
    print(f"\nEndurance eval at 4000 RPM in 5th: Target Gear = {target_gear}") # Expect 4

    # Plot points for one strategy
    endurance_strategy = manager.strategies.get("Endurance")
    if endurance_strategy:
        plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'transmission'))
        os.makedirs(plot_dir, exist_ok=True)
        # Need vehicle state for plotting speeds
        plot_state = {'gear_ratios': gears, 'wheel_radius': 0.2286, 'final_drive_ratio': 53/14.0}
        endurance_strategy.plot_shift_points(
            engine_rpm_range=np.linspace(1000, max_rpm, 100),
            vehicle_state=plot_state,
            save_path=os.path.join(plot_dir, "endurance_shift_points.png")
        )