import os
import sys
import traci

sumo_home = os.environ.get("SUMO_HOME")

if not sumo_home:
    raise EnvironmentError("SUMO_HOME is not set")

sumo_binary = os.path.join(sumo_home, "bin", "sumo-gui.exe")

sumo_config = "osm.sumocfg"

traci.start([
    sumo_binary,
    "-c", sumo_config,
    "--start"
])

print("SUMO connected successfully!")

for step in range(100):
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    print(
        f"Step: {step} | "
        f"Vehicles: {len(vehicles)}"
    )

traci.close()

print("Simulation finished.")