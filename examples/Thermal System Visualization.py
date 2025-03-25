"""
Formula Student Thermal System Visualization

This script demonstrates the thermal performance of different cooling system
configurations for a Formula Student racing car, providing visualizations
of thermal behavior under various operating conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from time import time

# Import modules from the Formula Student powertrain package
from ..kcl_fs_powertrain.thermal.cooling_system import (
    CoolingSystem, Radiator, RadiatorType, CoolingFan, FanType, WaterPump, PumpType, Thermostat
)
from ..kcl_fs_powertrain.thermal.rear_radiator import (
    RearRadiator, RearRadiatorDuct, RearRadiatorSystem, MountingPosition, DuctType
)
from ..kcl_fs_powertrain.thermal.side_pod import (
    SidePod, SidePodRadiator, SidePodSystem, DualSidePodSystem, SidePodType, RadiatorOrientation
)
from ..kcl_fs_powertrain.thermal.electric_compressor import (
    ElectricCompressor, CompressorControl, CoolingAssistSystem
)
from ..kcl_fs_powertrain.engine.engine_thermal import (
    ThermalConfig, EngineHeatModel, ThermalSimulation, CoolingSystem as EngineCoolingSystem
)
from ..kcl_fs_powertrain.utils.plotting import (
    set_plot_style, plot_thermal_performance, plot_thermal_comparison,
    plot_cooling_system_map, plot_endurance_results
)

# Create output directory for plots
output_dir = "thermal_results"
os.makedirs(output_dir, exist_ok=True)

# Set plot style
set_plot_style('clean')

def compare_radiator_configurations():
    """
    Compare different radiator configurations (side-pod vs. rear mounted).
    """
    print("Comparing radiator configurations...")
    
    # Create standard side-pod cooling system
    side_pod_system = create_formula_student_cooling_system()
    
    # Create rear-mounted cooling system
    rear_radiator_system = create_optimized_rear_radiator_system()
    
    # Create alternative cooling configurations
    side_pod_high_performance = create_cooling_optimized_side_pod_system()
    
    # Testing conditions
    vehicle_speeds = np.linspace(0, 30, 31)  # 0-30 m/s (0-108 km/h)
    coolant_temp = 90.0  # °C
    ambient_temp = 30.0  # °C - hot ambient conditions
    coolant_flow_rate = 50.0  # L/min
    
    # Run performance analysis
    sidepod_performance = side_pod_system.left_system.analyze_performance(
        vehicle_speed_range=vehicle_speeds,
        coolant_temp=coolant_temp,
        ambient_temp=ambient_temp,
        coolant_flow_rate=coolant_flow_rate/2  # Flow divided between two pods
    )
    
    rear_performance = rear_radiator_system.analyze_performance(
        vehicle_speed_range=vehicle_speeds,
        coolant_temp=coolant_temp,
        ambient_temp=ambient_temp,
        coolant_flow_rate=coolant_flow_rate
    )
    
    high_perf_sidepod = side_pod_high_performance.left_system.analyze_performance(
        vehicle_speed_range=vehicle_speeds,
        coolant_temp=coolant_temp,
        ambient_temp=ambient_temp,
        coolant_flow_rate=coolant_flow_rate/2  # Flow divided between two pods
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot heat rejection
    plt.subplot(2, 1, 1)
    plt.plot(vehicle_speeds, sidepod_performance['heat_rejections']/1000, 'b-', linewidth=2, 
             label='Standard Side Pod (per pod)')
    plt.plot(vehicle_speeds, rear_performance['heat_rejections']/1000, 'r-', linewidth=2, 
             label='Rear Radiator')
    plt.plot(vehicle_speeds, high_perf_sidepod['heat_rejections']/1000, 'g-', linewidth=2, 
             label='High-Performance Side Pod (per pod)')
    
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Heat Rejection (kW)')
    plt.title('Heat Rejection Capability vs. Vehicle Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot airflow
    plt.subplot(2, 1, 2)
    plt.plot(vehicle_speeds, sidepod_performance['airflows'], 'b-', linewidth=2, 
             label='Standard Side Pod (per pod)')
    plt.plot(vehicle_speeds, rear_performance['airflows'], 'r-', linewidth=2, 
             label='Rear Radiator')
    plt.plot(vehicle_speeds, high_perf_sidepod['airflows'], 'g-', linewidth=2, 
             label='High-Performance Side Pod (per pod)')
    
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Airflow (m³/s)')
    plt.title('Radiator Airflow vs. Vehicle Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radiator_configuration_comparison.png"), dpi=300)
    plt.close()
    
    # Calculate total system capabilities (both side pods combined vs. rear)
    sidepod_total_heat = sidepod_performance['heat_rejections'] * 2  # Two pods
    high_perf_total_heat = high_perf_sidepod['heat_rejections'] * 2  # Two pods
    
    # Plot total cooling capacity comparison
    plt.figure(figsize=(10, 6))
    plt.plot(vehicle_speeds, sidepod_total_heat/1000, 'b-', linewidth=2, 
             label='Standard Side Pods (total)')
    plt.plot(vehicle_speeds, rear_performance['heat_rejections']/1000, 'r-', linewidth=2, 
             label='Rear Radiator')
    plt.plot(vehicle_speeds, high_perf_total_heat/1000, 'g-', linewidth=2, 
             label='High-Performance Side Pods (total)')
    
    # Add reference line for typical engine heat rejection
    typical_heat = np.full_like(vehicle_speeds, 25)  # 25 kW typical heat rejection
    plt.plot(vehicle_speeds, typical_heat, 'k--', label='Typical Engine Heat Rejection')
    
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Heat Rejection (kW)')
    plt.title('Total Cooling System Heat Rejection')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_cooling_capacity_comparison.png"), dpi=300)
    plt.close()
    
    print("Radiator configuration comparison complete.")
    return sidepod_performance, rear_performance, high_perf_sidepod

def analyze_low_speed_cooling():
    """
    Analyze cooling system performance at low speeds with electric compressor assist.
    """
    print("Analyzing low-speed cooling performance...")
    
    # Create base cooling system
    cooling_system = create_formula_student_cooling_system()
    
    # Create cooling system with electric compressor assist
    cooling_assist = create_high_performance_cooling_assist_system()
    
    # Test conditions - static and low-speed conditions
    low_speeds = np.linspace(0, 10, 21)  # 0-10 m/s (0-36 km/h)
    coolant_temp = 85.0  # °C - starting coolant temp
    ambient_temp = 35.0  # °C - very hot ambient conditions
    coolant_flow_rate = 50.0  # L/min
    
    # Storage for results
    standard_temps = []
    assist_temps = []
    
    # Run simplified simulations for overheating scenario
    heat_input = 20000  # W (20kW heat input from engine)
    
    for speed in low_speeds:
        # Standard cooling - both side pods
        left_heat = cooling_system.left_system.calculate_heat_rejection(
            coolant_temp=coolant_temp, 
            ambient_temp=ambient_temp, 
            coolant_flow_rate=coolant_flow_rate/2, 
            vehicle_speed=speed
        )
        
        right_heat = cooling_system.right_system.calculate_heat_rejection(
            coolant_temp=coolant_temp, 
            ambient_temp=ambient_temp, 
            coolant_flow_rate=coolant_flow_rate/2, 
            vehicle_speed=speed
        )
        
        total_heat = left_heat + right_heat
        
        # Net heat into system
        net_heat = heat_input - total_heat
        
        # Calculate temperature effect (simplified)
        # Positive net heat means overheating
        standard_temps.append(coolant_temp + net_heat * 0.01)  # Arbitrary scaling
        
        # Cooling with electric compressor assist
        # Update system state
        cooling_assist.update_system(
            coolant_temp=coolant_temp, 
            vehicle_speed=speed, 
            engine_load=0.5, 
            dt=0.1
        )
        
        # Get supplementary airflow
        extra_airflow = cooling_assist.calculate_supplementary_airflow()
        
        # Calculate heat rejection with assist (approximate)
        # We need to translate the extra airflow into additional cooling capacity
        # This is a simplified approach - in practice would need more detailed modeling
        extra_cooling = extra_airflow * 2000  # Simplified conversion of airflow to cooling W
        assisted_total_heat = total_heat + extra_cooling
        
        # Net heat with assist
        net_heat_assist = heat_input - assisted_total_heat
        
        # Calculate temperature effect with assist
        assist_temps.append(coolant_temp + net_heat_assist * 0.01)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot temperatures
    plt.subplot(2, 1, 1)
    plt.plot(low_speeds, standard_temps, 'r-', linewidth=2, label='Standard Cooling')
    plt.plot(low_speeds, assist_temps, 'b-', linewidth=2, label='With Electric Compressor Assist')
    
    # Add warning/critical temperature lines
    warning_temp = 100
    critical_temp = 110
    plt.axhline(y=warning_temp, color='orange', linestyle='--', label='Warning Temperature')
    plt.axhline(y=critical_temp, color='red', linestyle='--', label='Critical Temperature')
    
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Coolant Temperature (°C)')
    plt.title('Coolant Temperature vs. Vehicle Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot supplementary airflow from cooling assist
    plt.subplot(2, 1, 2)
    
    # Calculate airflow at each speed
    supplementary_airflow = []
    for speed in low_speeds:
        cooling_assist.update_system(
            coolant_temp=coolant_temp, 
            vehicle_speed=speed, 
            engine_load=0.5, 
            dt=0.1
        )
        supplementary_airflow.append(cooling_assist.calculate_supplementary_airflow())
    
    plt.plot(low_speeds, supplementary_airflow, 'g-', linewidth=2)
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Supplementary Airflow (m³/s)')
    plt.title('Electric Compressor Supplementary Airflow')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "low_speed_cooling_performance.png"), dpi=300)
    plt.close()
    
    # Create cooling map visualization
    # This shows how low speed cooling is enhanced by the electric compressor
    speed_grid, temp_grid = np.meshgrid(low_speeds, np.linspace(80, 105, 20))
    cooling_map = np.zeros_like(speed_grid)
    assist_map = np.zeros_like(speed_grid)
    
    for i, temp in enumerate(np.linspace(80, 105, 20)):
        for j, speed in enumerate(low_speeds):
            # Standard cooling - both side pods
            left_heat = cooling_system.left_system.calculate_heat_rejection(
                coolant_temp=temp, 
                ambient_temp=ambient_temp, 
                coolant_flow_rate=coolant_flow_rate/2, 
                vehicle_speed=speed
            )
            
            right_heat = cooling_system.right_system.calculate_heat_rejection(
                coolant_temp=temp, 
                ambient_temp=ambient_temp, 
                coolant_flow_rate=coolant_flow_rate/2, 
                vehicle_speed=speed
            )
            
            cooling_map[i, j] = left_heat + right_heat
            
            # With assist
            cooling_assist.update_system(
                coolant_temp=temp, 
                vehicle_speed=speed, 
                engine_load=0.5, 
                dt=0.1
            )
            
            extra_airflow = cooling_assist.calculate_supplementary_airflow()
            extra_cooling = extra_airflow * 2000  # Simplified conversion
            
            assist_map[i, j] = (left_heat + right_heat) + extra_cooling
    
    # Plot cooling capability map
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    contour = plt.contourf(speed_grid, temp_grid, cooling_map/1000, 15, cmap='cool')
    plt.colorbar(contour, label='Cooling Capacity (kW)')
    plt.contour(speed_grid, temp_grid, cooling_map/1000, 10, colors='k', alpha=0.3)
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Coolant Temperature (°C)')
    plt.title('Standard Cooling Capacity')
    
    plt.subplot(1, 2, 2)
    contour = plt.contourf(speed_grid, temp_grid, assist_map/1000, 15, cmap='cool')
    plt.colorbar(contour, label='Cooling Capacity (kW)')
    plt.contour(speed_grid, temp_grid, assist_map/1000, 10, colors='k', alpha=0.3)
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Coolant Temperature (°C)')
    plt.title('Cooling Capacity with Electric Compressor Assist')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cooling_capacity_map.png"), dpi=300)
    plt.close()
    
    print("Low-speed cooling analysis complete.")

def simulate_endurance_event():
    """
    Simulate thermal performance during an endurance event.
    """
    print("Simulating endurance event thermal performance...")
    
    # Create cooling system configuration
    cooling_system = create_formula_student_cooling_system()
    
    # Initialize thermal configuration and model
    thermal_config = ThermalConfig()
    engine_model = EngineHeatModel(thermal_config)
    engine_cooling = EngineCoolingSystem(thermal_config)
    
    # Create thermal simulation
    simulation = ThermalSimulation(engine_model, engine_cooling)
    
    # Endurance event simulation parameters
    total_laps = 20
    lap_distance = 1000  # m
    lap_time_avg = 75  # seconds
    lap_time_array = np.random.normal(lap_time_avg, 2, total_laps)  # Some variation in lap times
    
    # Speed profiles
    # Create a representative speed profile for a lap
    points_per_lap = 300
    distance_points = np.linspace(0, lap_distance, points_per_lap)
    
    # Create a sinusoidal speed profile with straights and corners
    base_speed = 15  # m/s average speed
    speed_variation = 10  # m/s variation
    
    # Simplified speed profile with several corners
    speed_profile = base_speed + speed_variation * np.sin(distance_points * 0.02) * 0.5 + \
                  speed_variation * np.sin(distance_points * 0.01) * 0.7
    
    # Ensure all speeds are positive and reasonable
    speed_profile = np.maximum(speed_profile, 5)
    
    # Engine load profile (approximated based on speed)
    load_profile = 0.5 + 0.4 * speed_profile / np.max(speed_profile)
    
    # Temperature and cooling data storage
    time_points = []
    engine_temps = []
    coolant_temps = []
    oil_temps = []
    vehicle_speeds = []
    heat_rejections = []
    ambient_temp = 30.0  # Hot day
    
    # Interpolation functions for speed and load
    def get_speed_at_distance(d):
        # Normalize distance within a lap
        d_norm = d % lap_distance
        idx = int(d_norm / lap_distance * (points_per_lap - 1))
        return speed_profile[idx]
    
    def get_load_at_distance(d):
        # Normalize distance within a lap
        d_norm = d % lap_distance
        idx = int(d_norm / lap_distance * (points_per_lap - 1))
        return load_profile[idx]
    
    # Run simulation
    current_time = 0
    total_distance = 0
    dt = 0.1  # seconds
    
    # Initialize temperatures
    engine_temp = 80.0  # Starting engine temp
    coolant_temp = 75.0
    oil_temp = 70.0
    
    for lap in range(1, total_laps + 1):
        # Each lap
        lap_time = lap_time_array[lap-1]
        steps_per_lap = int(lap_time / dt)
        
        for step in range(steps_per_lap):
            # Calculate progress through lap
            progress = step / steps_per_lap
            distance_in_lap = progress * lap_distance
            total_distance = (lap - 1) * lap_distance + distance_in_lap
            
            # Get vehicle conditions at this point
            vehicle_speed = get_speed_at_distance(total_distance)
            engine_load = get_load_at_distance(total_distance)
            
            # Simple engine power calculation
            engine_rpm = 3000 + engine_load * 8000  # RPM between 3000 and 11000
            engine_torque = 40 + engine_load * 30  # Nm between 40 and 70
            
            # Calculate heat generation
            # Engine power in W
            engine_power = engine_torque * engine_rpm * 2 * np.pi / 60
            
            # Assuming 30% efficiency, 70% of fuel energy becomes heat
            heat_generated = engine_power * 0.7 / 0.3
            
            # Calculate cooling capacity
            # Both side pods - at coolant temp, ambient, vehicle speed
            left_heat = cooling_system.left_system.calculate_heat_rejection(
                coolant_temp=coolant_temp, 
                ambient_temp=ambient_temp, 
                coolant_flow_rate=55.0/2,  # Divided between pods
                vehicle_speed=vehicle_speed
            )
            
            right_heat = cooling_system.right_system.calculate_heat_rejection(
                coolant_temp=coolant_temp, 
                ambient_temp=ambient_temp, 
                coolant_flow_rate=55.0/2,  # Divided between pods 
                vehicle_speed=vehicle_speed
            )
            
            total_cooling = left_heat + right_heat
            
            # Update temperatures (simplified model)
            # Heat balance = heat generated - heat rejected
            net_heat = heat_generated - total_cooling
            
            # Temperature change rate depends on thermal mass
            # These are simplifications - in a real model we'd use the actual thermal model
            engine_temp_change = net_heat * 0.0001  # Arbitrary scaling based on thermal mass
            engine_temp += engine_temp_change * dt
            
            # Coolant and oil temperatures follow engine temperature with lag
            coolant_temp += (engine_temp - coolant_temp) * 0.05 * dt
            oil_temp += (engine_temp - oil_temp) * 0.03 * dt
            
            # Store data
            time_points.append(current_time)
            engine_temps.append(engine_temp)
            coolant_temps.append(coolant_temp)
            oil_temps.append(oil_temp)
            vehicle_speeds.append(vehicle_speed)
            heat_rejections.append(total_cooling)
            
            # Increment time
            current_time += dt
    
    # Prepare data for plotting
    thermal_data = {
        'time': time_points,
        'engine_temp': engine_temps,
        'coolant_temp': coolant_temps,
        'oil_temp': oil_temps,
        'vehicle_speed': vehicle_speeds,
        'heat_rejection': heat_rejections,
        'ambient_temp': [ambient_temp] * len(time_points)
    }
    
    # Plot thermal performance
    fig = plot_thermal_performance(thermal_data, 
                               title='Formula Student Endurance Event - Thermal Performance',
                               save_path=os.path.join(output_dir, "endurance_thermal_performance.png"))
    
    # Create endurance summary visualization
    plt.figure(figsize=(12, 10))
    
    # Plot temperatures over laps
    lap_markers = np.cumsum(lap_time_array)
    lap_indices = [np.argmin(np.abs(np.array(time_points) - marker)) for marker in lap_markers]
    
    lap_eng_temps = [engine_temps[i] for i in lap_indices]
    lap_cool_temps = [coolant_temps[i] for i in lap_indices]
    lap_oil_temps = [oil_temps[i] for i in lap_indices]
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, total_laps+1), lap_eng_temps, 'ro-', label='Engine')
    plt.plot(range(1, total_laps+1), lap_cool_temps, 'bo-', label='Coolant')
    plt.plot(range(1, total_laps+1), lap_oil_temps, 'go-', label='Oil')
    
    plt.axhline(y=100, color='orange', linestyle='--', label='Warning')
    plt.axhline(y=110, color='red', linestyle='--', label='Critical')
    
    plt.xlabel('Lap Number')
    plt.ylabel('Temperature (°C)')
    plt.title('End-of-Lap Temperatures')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot temperature vs. speed scatter
    plt.subplot(2, 2, 2)
    plt.scatter(vehicle_speeds, coolant_temps, c=engine_temps, cmap='hot', alpha=0.5)
    plt.colorbar(label='Engine Temperature (°C)')
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Coolant Temperature (°C)')
    plt.title('Temperature vs. Speed Relationship')
    plt.grid(True, alpha=0.3)
    
    # Plot heat rejection histogram
    plt.subplot(2, 2, 3)
    plt.hist(heat_rejections, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Heat Rejection (W)')
    plt.ylabel('Frequency')
    plt.title('Heat Rejection Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot temperature vs. time with lap markers
    plt.subplot(2, 2, 4)
    plt.plot(time_points, engine_temps, 'r-', alpha=0.5, label='Engine')
    plt.plot(time_points, coolant_temps, 'b-', alpha=0.5, label='Coolant')
    
    # Add lap markers
    for i, lap_time in enumerate(lap_markers):
        if i % 2 == 0:  # Plot every other lap marker to avoid clutter
            plt.axvline(x=lap_time, color='gray', linestyle='--', alpha=0.3)
            plt.text(lap_time, max(engine_temps) + 2, f'L{i+1}', 
                     fontsize=8, horizontalalignment='center')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs. Time with Lap Markers')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "endurance_thermal_summary.png"), dpi=300)
    plt.close()
    
    print("Endurance simulation complete.")
    return thermal_data

def analyze_hot_weather_performance():
    """
    Analyze cooling system performance in hot weather conditions.
    """
    print("Analyzing hot weather performance...")
    
    # Create different cooling system configurations
    standard_system = create_formula_student_cooling_system()
    cooling_optimized = create_cooling_optimized_side_pod_system()
    
    # Create cooling system with electric compressor assist
    electric_assist = create_high_performance_cooling_assist_system()
    
    # Testing conditions - various ambient temperatures and vehicle speeds
    ambient_temps = np.linspace(20, 45, 6)  # 20°C to 45°C
    vehicle_speeds = np.linspace(0, 25, 6)  # 0 to 25 m/s
    
    coolant_temp = 90.0  # °C - fixed coolant temperature
    engine_load = 0.8  # High engine load
    engine_heat_input = 30000  # W (30kW heat input - high load)
    
    # Create heatmap data
    standard_margin = np.zeros((len(ambient_temps), len(vehicle_speeds)))
    optimized_margin = np.zeros_like(standard_margin)
    assist_margin = np.zeros_like(standard_margin)
    
    # Calculate cooling margin (cooling capacity - heat input)
    for i, ambient in enumerate(ambient_temps):
        for j, speed in enumerate(vehicle_speeds):
            # Standard system
            left_heat = standard_system.left_system.calculate_heat_rejection(
                coolant_temp=coolant_temp, 
                ambient_temp=ambient, 
                coolant_flow_rate=50.0/2, 
                vehicle_speed=speed
            )
            
            right_heat = standard_system.right_system.calculate_heat_rejection(
                coolant_temp=coolant_temp, 
                ambient_temp=ambient, 
                coolant_flow_rate=50.0/2, 
                vehicle_speed=speed
            )
            
            standard_margin[i, j] = (left_heat + right_heat) - engine_heat_input
            
            # Optimized system
            left_heat = cooling_optimized.left_system.calculate_heat_rejection(
                coolant_temp=coolant_temp, 
                ambient_temp=ambient, 
                coolant_flow_rate=50.0/2, 
                vehicle_speed=speed
            )
            
            right_heat = cooling_optimized.right_system.calculate_heat_rejection(
                coolant_temp=coolant_temp, 
                ambient_temp=ambient, 
                coolant_flow_rate=50.0/2, 
                vehicle_speed=speed
            )
            
            optimized_margin[i, j] = (left_heat + right_heat) - engine_heat_input
            
            # With electric assist
            electric_assist.update_system(
                coolant_temp=coolant_temp, 
                vehicle_speed=speed, 
                engine_load=engine_load, 
                dt=0.1
            )
            
            extra_airflow = electric_assist.calculate_supplementary_airflow()
            extra_cooling = extra_airflow * 2000  # Simplified conversion
            
            # Use standard system cooling + assist
            assist_margin[i, j] = standard_margin[i, j] + extra_cooling
    
    # Plot cooling margin heatmaps
    # Convert to kW for better readability
    standard_margin_kw = standard_margin / 1000
    optimized_margin_kw = optimized_margin / 1000
    assist_margin_kw = assist_margin / 1000
    
    # Get min/max values for consistent color scaling
    vmin = min(np.min(standard_margin_kw), np.min(optimized_margin_kw), np.min(assist_margin_kw))
    vmax = max(np.max(standard_margin_kw), np.max(optimized_margin_kw), np.max(assist_margin_kw))
    
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    X, Y = np.meshgrid(vehicle_speeds, ambient_temps)
    c1 = plt.contourf(X, Y, standard_margin_kw, 20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(c1, label='Cooling Margin (kW)')
    plt.contour(X, Y, standard_margin_kw, levels=[0], colors='k', linestyles='-', linewidths=2)
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.title('Standard Cooling System')
    
    plt.subplot(1, 3, 2)
    c2 = plt.contourf(X, Y, optimized_margin_kw, 20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(c2, label='Cooling Margin (kW)')
    plt.contour(X, Y, optimized_margin_kw, levels=[0], colors='k', linestyles='-', linewidths=2)
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.title('Optimized Cooling System')
    
    plt.subplot(1, 3, 3)
    c3 = plt.contourf(X, Y, assist_margin_kw, 20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(c3, label='Cooling Margin (kW)')
    plt.contour(X, Y, assist_margin_kw, levels=[0], colors='k', linestyles='-', linewidths=2)
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.title('Standard + Electric Assist')
    
    plt.suptitle('Cooling Margin in Hot Weather Conditions (30kW Heat Load)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "hot_weather_performance.png"), dpi=300)
    plt.close()
    
    # Calculate the maximum ambient temperature each system can handle
    max_temps = {
        'Standard': [],
        'Optimized': [],
        'With Assist': []
    }
    
    for j, speed in enumerate(vehicle_speeds):
        # Find maximum ambient temp where margin is still positive
        for i in range(len(ambient_temps) - 1):
            if standard_margin[i, j] >= 0 and standard_margin[i + 1, j] < 0:
                # Interpolate to find exact ambient temp where margin is zero
                interp = ambient_temps[i] + (ambient_temps[i + 1] - ambient_temps[i]) * \
                         (0 - standard_margin[i, j]) / (standard_margin[i + 1, j] - standard_margin[i, j])
                max_temps['Standard'].append(interp)
                break
            elif i == len(ambient_temps) - 2 and standard_margin[i + 1, j] >= 0:
                max_temps['Standard'].append(ambient_temps[i + 1])  # Can handle highest temp
            elif i == 0 and standard_margin[i, j] < 0:
                max_temps['Standard'].append(ambient_temps[i])  # Can't handle lowest temp
        
        # Repeat for optimized system
        for i in range(len(ambient_temps) - 1):
            if optimized_margin[i, j] >= 0 and optimized_margin[i + 1, j] < 0:
                interp = ambient_temps[i] + (ambient_temps[i + 1] - ambient_temps[i]) * \
                         (0 - optimized_margin[i, j]) / (optimized_margin[i + 1, j] - optimized_margin[i, j])
                max_temps['Optimized'].append(interp)
                break
            elif i == len(ambient_temps) - 2 and optimized_margin[i + 1, j] >= 0:
                max_temps['Optimized'].append(ambient_temps[i + 1])
            elif i == 0 and optimized_margin[i, j] < 0:
                max_temps['Optimized'].append(ambient_temps[i])
        
        # Repeat for assisted system
        for i in range(len(ambient_temps) - 1):
            if assist_margin[i, j] >= 0 and assist_margin[i + 1, j] < 0:
                interp = ambient_temps[i] + (ambient_temps[i + 1] - ambient_temps[i]) * \
                         (0 - assist_margin[i, j]) / (assist_margin[i + 1, j] - assist_margin[i, j])
                max_temps['With Assist'].append(interp)
                break
            elif i == len(ambient_temps) - 2 and assist_margin[i + 1, j] >= 0:
                max_temps['With Assist'].append(ambient_temps[i + 1])
            elif i == 0 and assist_margin[i, j] < 0:
                max_temps['With Assist'].append(ambient_temps[i])
    
    # Plot maximum operating ambient temperature
    plt.figure(figsize=(10, 6))
    plt.plot(vehicle_speeds, max_temps['Standard'], 'b-', linewidth=2, label='Standard Cooling')
    plt.plot(vehicle_speeds, max_temps['Optimized'], 'g-', linewidth=2, label='Optimized Cooling')
    plt.plot(vehicle_speeds, max_temps['With Assist'], 'r-', linewidth=2, label='With Electric Assist')
    
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Maximum Ambient Temperature (°C)')
    plt.title('Maximum Operating Ambient Temperature vs. Vehicle Speed')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_operating_temperature.png"), dpi=300)
    plt.close()
    
    print("Hot weather performance analysis complete.")

def run_all_simulations():
    """Run all thermal simulations and visualizations."""
    print("Running all Formula Student thermal simulations...")
    
    # Start timing
    start_time = time()
    
    # Compare radiator configurations
    sidepod_perf, rear_perf, high_perf = compare_radiator_configurations()
    
    # Analyze low-speed cooling
    analyze_low_speed_cooling()
    
    # Simulate endurance event
    thermal_data = simulate_endurance_event()
    
    # Analyze hot weather performance
    analyze_hot_weather_performance()
    
    # End timing
    end_time = time()
    print(f"All simulations completed in {end_time - start_time:.2f} seconds.")
    print(f"Results saved to '{output_dir}' directory.")

if __name__ == "__main__":
    # Import functions from the modules - for local testing
    # These will be defined in the powertrain package
    from ..kcl_fs_powertrain.thermal.cooling_system import create_formula_student_cooling_system
    from ..kcl_fs_powertrain.thermal.rear_radiator import create_optimized_rear_radiator_system
    from ..kcl_fs_powertrain.thermal.side_pod import create_cooling_optimized_side_pod_system
    from ..kcl_fs_powertrain.thermal.electric_compressor import create_high_performance_cooling_assist_system
    
    # Run all simulations
    run_all_simulations()