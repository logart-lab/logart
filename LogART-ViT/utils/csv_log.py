import csv
import os

def log_to_csv(csv_path, args_dict, result_dict):
    """
    Log experiment arguments and results to a CSV file.

    :param csv_path: Path to the CSV file.
    :param args_dict: Dictionary of selected experiment arguments.
    :param result_dict: Dictionary of experiment results (e.g., accuracy, loss).
    """
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(args_dict.keys()) + list(result_dict.keys())

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        row = {**args_dict, **result_dict}
        writer.writerow(row)
