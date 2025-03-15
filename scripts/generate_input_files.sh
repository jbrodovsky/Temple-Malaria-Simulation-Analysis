#!/bin/bash

populations=(10 50 100 150 1500 15000)
beta_values=(0.001 0.005 0.01 0.0125 0.015 0.02 0.03 0.04 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1)
treatment_access_rates=(0.55 0.65 0.75 0.85)
country="bfa"
zones=(1)
O5ADJUST=1.0

# Delete the "input_files" directory if it exists and create a new one
if [ -d "input_files" ]; then
    rm -rf "input_files"
fi
mkdir "input_files"

# Generate zone ASC files
for zone in "${zones[@]}"; do
    sed 's/#ZONE#/'"$zone"'/g' zone.asc > "input_files/$zone.asc"
done

# Generate population ASC files
for population in "${populations[@]}"; do
    sed 's/#POPULATION#/'"$population"'/g' population.asc > "input_files/$population.asc"
done

# Generate YML files for each combination of parameters
for population in "${populations[@]}" ; do
    for beta in "${beta_values[@]}" ; do
        for access in "${treatment_access_rates[@]}" ; do
            filename="input_files/$zone-$population-$access-$beta-$country.yml"
            echo "Preparing yml file for zone: ${zone} population: ${population} access: ${access} beta : ${beta} country: ${country}"
            sed 's/#BETA#/'"$beta"'/g' $country-calibration.yml > "$filename"
            sed -i 's/#POPULATION#/'"$population"'/g' "$filename"
            sed -i 's/#ACCESSU5#/'"$access"'/g' "$filename"
            sed -i 's/#ACCESSO5#/'"$(bc -l <<< $access*$O5ADJUST)"'/g' "$filename"
            sed -i 's/#ZONE#/'"$zone"'/g' "$filename"
        done
    done
done
