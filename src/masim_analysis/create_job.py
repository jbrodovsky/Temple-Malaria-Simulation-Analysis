"""
Generate a job file for MaSim commands.
"""

import argparse
from pathlib import Path
from masim_analysis.commands import Cluster, generate_job_file


def create_job() -> None:
    """CLI bindings for generate_job_file."""
    parser = argparse.ArgumentParser(description="Generate a job file for MaSim commands")
    parser.add_argument(
        "-c",
        "--commands",
        type=str,
        required=True,
        help="The file containing the MaSim commands to execute, ex: ./commands.txt",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="MyJob",
        help="The name of the job to submit to the cluster (default: MyJob)",
    )
    parser.add_argument(
        "-N",
        "--node",
        type=Cluster,
        choices=list(Cluster),
        default=Cluster.ONE,
        help="The cluster node to use (default: nd01)",
    )
    parser.add_argument(
        "-cpo",
        "--cores-override",
        type=int,
        default=None,
        help="Override the number of cores per node (default: None, which uses maximum available)",
    )
    parser.add_argument(
        "-t",
        "--time-override",
        type=int,
        default=48,
        help="Override the wall time in hours (default: 48)",
    )
    parser.add_argument(
        "-o",
        "--std-output-location",
        type=str,
        default=Path("."),
        help="Standard output file location (default: None, which writes to ./<job_name>.output)",
    )
    parser.add_argument(
        "-e",
        "--std-error-location",
        type=str,
        default=Path("."),
        help="Standard error file location (default: None, which writes to ./<job_name>.error)",
    )
    parser.add_argument(
        "-m",
        "--email",
        type=str,
        default=None,
        help="Email address for job notifications (default: None)",
    )

    args = parser.parse_args()
    generate_job_file(
        args.commands,
        node=args.node,
        job_name=args.name,
        cores_override=args.cores_override,
        time_override=args.time_override,
        std_output_location=args.std_output_location,
        std_error_location=args.std_error_location,
        email=args.email,
    )


if __name__ == "__main__":
    create_job()
