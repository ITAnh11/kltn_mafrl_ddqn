# common.py
import copy
import csv
import os
import random
import numpy as np


class CSVLogger:
    """CSV logger for training metrics."""

    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.file = open(filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
