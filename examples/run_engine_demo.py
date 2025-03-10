#!/usr/bin/env python3
"""
Engine Demonstration Script

This script automates the process of setting up and running the
Formula Student engine simulation program.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def check_dependencies():
    """Check and install required dependencies."""
    required_packages = [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
        "CoolProp>=6.4.0",
        "plotly>=5.3.0"
    ]
    
    print("Checking dependencies...")
    
    # Check if each package is installed
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            importlib.import_module(package_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"⨯ {package_name} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package_name} has been installed")


def create_directories():
    """Create necessary directories for the engine program."""
    directories = [
        "data/output/engine",
        "data/output/tracks",
        "configs/engine"
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory} directory is ready")


def ensure_config_file():
    """Ensure the engine configuration file exists."""
    config_path = "configs/engine/cbr600f4i.yaml"
    
    # Check if the config file exists
    if os.path.exists(config_path):
        print(f"\n✓ Engine configuration file already exists at {config_path}")
        return
    
    # Create a basic config file if it doesn't exist
    print(f"\nCreating engine configuration file at {config_path}...")
    
    config_content = """# Honda CBR600F4i Engine Configuration
make: 'Honda'
model: 'CBR600F4i'
year: 2004
displacement_cc: 599
cylinders: 4
configuration: 'Inline'
compression_ratio: 12.0
bore_mm: 67.0
stroke_mm: 42.5
valves_per_cylinder: 4
valve_train_type: 'DOHC'
intake_open_btdc: 25
intake_close_abdc: 55
exhaust_open_bbdc: 25
exhaust_close_atdc: 10
max_power_hp: 110
max_power_rpm: 12500
max_torque_nm: 65
max_torque_rpm: 10500
redline_rpm: 14000
idle_rpm: 1300
dry_weight_kg: 57  # Engine weight estimate
cooling_type: 'Liquid'
fuel_system: 'PGM-FI'
fuel_type: 'Gasoline'
ignition_type: 'Digital transistorized'

# FS-specific modifications
fs_modifications:
  exhaust: 'Custom header and exhaust system'
  intake: 'Velocity stacks and custom airbox'
  ecu: 'Reprogrammed for E85 fuel'
  weight_reduction: 'Removed starter motor, lights, and other non-essentials'
  air_filter: 'High flow air filter'
  fuel_injectors: 'Higher flow rate injectors for E85'
fs_weight_reduction_kg: 5.0
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Engine configuration file created at {config_path}")


def run_engine_demo():
    """Run a basic demonstration of the engine program."""
    print("\nRunning Engine Demonstration...")
    
    try:
        # Try to import from the installed package first
        try:
            from kcl_fs_powertrain.engine import MotorcycleEngine, TorqueCurve, FuelType, FuelProperties
            from kcl_fs_powertrain.engine import ThermalConfig, CoolingSystem, EngineHeatModel
            print("✓ Successfully imported engine modules from installed package")
        except ImportError:
            # If not installed, add the current directory to the path
            sys.path.insert(0, os.path.abspath('.'))
            from kcl_fs_powertrain.engine import MotorcycleEngine, TorqueCurve, FuelType, FuelProperties
            from kcl_fs_powertrain.engine import ThermalConfig, CoolingSystem, EngineHeatModel
            print("✓ Successfully imported engine modules from local directory")
        
        # Create engine using config file
        config_path = os.path.join("configs", "engine", "cbr600f4i.yaml")
        engine = MotorcycleEngine(config_path=config_path)
        
        # Display engine specifications
        print("\nEngine Specifications:")
        specs = engine.get_engine_specs()
        for key, value in specs.items():
            print(f"  {key}: {value}")
        
        # Generate output directory path
        output_dir = os.path.join("data", "output", "engine")
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot and save performance curves
        curve_path = os.path.join(output_dir, "performance_curves.png")
        print(f"\nGenerating performance curves and saving to {curve_path}")
        engine.plot_performance_curves(save_path=curve_path)
        
        # Create torque curve analysis
        print("\nPerforming torque curve analysis...")
        torque_curve = TorqueCurve()
        torque_curve.load_from_engine(engine)
        
        # Find peak values
        peak_torque_rpm, peak_torque = torque_curve.find_peak_torque()
        peak_power_rpm, peak_power = torque_curve.find_peak_power()
        peak_power_hp = peak_power * 1.34102  # Convert to HP
        
        print(f"  Peak Torque: {peak_torque:.1f} Nm @ {peak_torque_rpm:.0f} RPM")
        print(f"  Peak Power: {peak_power_hp:.1f} hp @ {peak_power_rpm:.0f} RPM")
        
        # Create modified curve for E85
        print("\nSimulating E85 fuel conversion...")
        modified_curve = TorqueCurve(torque_curve.rpm_points.copy(), torque_curve.torque_values.copy())
        modified_curve.modify_for_e85()
        
        # Plot and save comparison
        comparison_path = os.path.join(output_dir, "e85_comparison.png")
        print(f"Generating E85 comparison and saving to {comparison_path}")
        torque_curve.plot_comparison(
            modified_curve, 
            labels=('Stock', 'E85'),
            title='E85 Fuel Conversion Effect',
            save_path=comparison_path
        )
        
        # Create thermal model
        print("\nRunning thermal simulation...")
        thermal_config = ThermalConfig()
        cooling_system = CoolingSystem(thermal_config)
        heat_model = EngineHeatModel(thermal_config, engine)
        
        # Print thermal model state
        temps = heat_model.get_temperature_state()
        print(f"  Initial temperatures - Engine: {temps['engine']:.1f}°C, Oil: {temps['oil']:.1f}°C, Coolant: {temps['coolant']:.1f}°C")
        
        # Create fuel properties
        print("\nAnalyzing fuel properties...")
        e85 = FuelProperties(FuelType.E85)
        gasoline = FuelProperties(FuelType.GASOLINE)
        
        # Compare fuels
        comparison = e85.compare_with(gasoline)
        print("  E85 vs Gasoline comparison:")
        for key, value in comparison.items():
            print(f"    {key}: {value:.2f}x")
        
        print("\n✓ Engine demonstration completed successfully!")
        print(f"\nResults have been saved to the {output_dir} directory.")
        print("You can now explore the engine module with your own simulations.")
        
    except Exception as e:
        print(f"\n⨯ Error running engine demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main function to run the entire setup and demonstration."""
    print("=" * 80)
    print("Formula Student Engine Program Setup and Demonstration")
    print("=" * 80)
    
    # Check dependencies
    check_dependencies()
    
    # Create directories
    create_directories()
    
    # Ensure config file exists
    ensure_config_file()
    
    # Run engine demonstration
    success = run_engine_demo()
    
    if success:
        print("\n" + "=" * 80)
        print("Setup and demonstration completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Setup completed, but demonstration encountered errors.")
        print("=" * 80)


if __name__ == "__main__":
    main()