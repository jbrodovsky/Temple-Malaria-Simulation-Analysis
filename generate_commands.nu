# This script is used to generate the commands.txt file used for torque-launch
# The commands.txt file is used to launch the jobs on the cluster as such:
# ```shell
# torque-launch commands.txt
# ```
# The MaSim simulation needs a input .yml file that describes the simulation
# parameters and events. The simulation binary itself cannot run multiple strategies
# in parallel or multiple repetitions. The simulation will run through one sequence
# of events (which may or may not contain multiple strategies) and then stop.
# In order to do actual comparative science, we need to run multiple repetitions
# of the simulation for each strategy. It is then beneficial to run these 
# repetitions in parallel on a cluster. 
#
# 

let num_repeats = 10;
let strategies = ["status_quo", 
                                "AL5",
                                "ASAQ", 
                                "DHA-PPQ", 
                                "AL25-ASAQ75",
                                "AL50-ASAQ50",
                                "AL75-ASAQ25",
                                "AL25-DHAPPQ75",
                                "AL50-DHAPPQ50",
                                "AL75-DHAPPQ25",
                                "ASAQ25-DHAPPQ75",
                                "ASAQ50-DHAPPQ50",
                                "ASAQ75-DHAPPQ25",];