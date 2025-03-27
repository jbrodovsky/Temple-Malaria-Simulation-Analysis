# Batch processing calibration script for local runs using Nushell
# This script is intended to be run from the command line and takes the filepath to the 
# MaSim *_cmds.txt file use for calibrations as an argument. This script is intended to
# replace the torque-launch command used in the Owl's Nest job file.

# Notes: nushell does not support the notion (like in bash) of eval <string>. Instead,
# the "command string" should be mannually written as a closure and then executed using
# the "do" command. This is a bit cumbersome, but it is what it is. This script should
# effectively recreate the functionality of both torque-launch and the "generate_commands"
# section in the notebook (copied below for reference).
use std/log

let cores = sys cpu | length;
#let total_ram = sys mem | get total | into int | $in / (1024 * 1024 * 1024);  # Convert to GB
# Input parameters
let country_code = "moz"
let populations = [10, 25] #, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000, 20000]
let access_rates = [0.50, 0.55] #, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
let betas = [0.001, 0.005] #, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
let reps = 2
# create output directories
mkdir $"./output/($country_code)/calibration"
# Initialize logging
$"(date now) | Starting new calibration\n" | save --append $"./output/($country_code)/calibration/calibration_log.txt"
let cmds = create_cmds $country_code $populations $access_rates $betas $reps
print $"Running jobs with a maximum of ($cores) jobs at a time..."
$cmds | par-each --threads $cores { |job| do $job }
print "All jobs are done!"
$"(date now) | All jobs are done!\n" | save --append $"./output/($country_code)/calibration/calibration_log.txt"
#########

def create_cmds [country: string, populations: list<int>, access_rates: list<float>, betas: list<float>, reps: int] {
    print $"Generating calibration commands for ($country) with ($populations | length) populations, ($access_rates | length) access rates, ($betas | length) betas, and ($reps) repetitions.";
    let cmds = ($populations | each { |pop|
        ($access_rates | each { |access|
            ($betas | each { |beta|
                (1..$reps | each { |j|
                   {
                    let print_str = $"Running ($country) calibration for ($pop) with access rate ($access) and beta ($beta) for repetition ($j)..."                    
                    log info $print_str
                    $"(date now) | ($print_str)\n" | save --append $"./output/($country)/calibration/calibration_log.txt"                    
                    ./bin/MaSim -i ./conf/($country)/calibration/cal_($pop)_($access)_($beta).yml -o ./output/($country)/calibration/cal_($pop)_($access)_($beta)_ -r SQLiteDistrictReporter -j ($j);
                }
                })
            })
        })
    } | flatten | flatten | flatten);     
    return $cmds;
}