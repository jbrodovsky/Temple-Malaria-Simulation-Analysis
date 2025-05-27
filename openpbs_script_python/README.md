### Note: Testing on `nd04`

To set up your environment for testing on `nd04`, follow these steps:

1. **Extract the script file**
    ```plaintext
    openpbs_script_malasim
    ├── bin
    ├── log
    ├── script
    └── README.md (this file)
    ```

2. **Navigate to the `script` folder**:
    ```bash
    cd script
    ```

3. **Generate submit script**:  
    ```bash
    chmod +x gen_submit_pbs.sh
    ./gen_submit_pbs.sh 7
    ```
    where `7` is number of jobs run in parallel.

4. **Submit your job**:
    ```bash
    qsub submit.pbs
    ```

    You will see a list of output and error files in `log` folder