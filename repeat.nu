print "Running MaSim with multiple strategies and repeats...";
let num_repeats = 10;
let strategies = ["status_quo", 
                                "AL5",
                                "ASAQ", 
                                "DHA-PPQ", 
                                "AL25-ASAQ75",
                                "AL50-ASAQ50",
                                "AL75-ASAQ25",
                                "AL25-DHA-PPQ75",
                                "AL50-DHA-PPQ50",
                                "AL75-DHA-PPQ25",
                                "ASAQ25-DHA-PPQ75",
                                "ASAQ50-DHA-PPQ50",
                                "ASAQ75-DHA-PPQ25",];

# Estimate memory constraints (Adjust these based on your system)
let cores = sys cpu | length;
let total_ram = sys mem | get total | into int | $in / (1024 * 1024 * 1024);  # Convert to GB
let mem_per_job = 4.5;  # Estimated RAM usage per job (in GB, adjust as needed)
mut max_jobs = ($total_ram // $mem_per_job) | into int;  # Max jobs based on RAM;
if $max_jobs > $cores { $max_jobs = $cores; }
# Print the estimated number of jobs and ask the user if they would like to change it or continue
print $"Estimated maximum number of jobs: ($max_jobs)";
let response = input "Would you like to change the number of jobs? (y/n)";
if $response == "y" {
    let new_max_jobs = input "Enter the new maximum number of jobs: ";
    $max_jobs = $new_max_jobs | into int;
}
# Generate the list of jobs
let jobs = ($strategies | each { |strategy|
    let input_file = $"./input_($strategy).yml";
    0..$num_repeats | each { |i|
        let output_file = $"./rwanda_($strategy)_($i)_";
        { 
            print $"Running ($strategy) repeat ($i)...";
            ./MaSim -i $input_file -r SQLiteDistrictReporter -o $output_file;
        }
    }
} | flatten)

# Run jobs with a RAM-based thread limit
print $jobs
print $"Running jobs with a maximum of ($max_jobs) jobs at a time..."
$jobs | par-each --threads $max_jobs { |job| do $job }

# Tell me that everything is done
print "All jobs are done!"
