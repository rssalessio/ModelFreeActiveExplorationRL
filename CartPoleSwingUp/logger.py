import os
from typing import List

class Logger(object):
    def __init__(self, file_name: str, columns: List[str], delimiter: str=','):
        """
        :param file_name: path to the file
        :param columns: columns in the first row of the file. Needs to be a list of strings
        :param delimite: delimiter between values
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.file_name = file_name
        self.columns = columns
        self.delimiter = delimiter
        self.num_columns = len(columns)
        self.file = open(self.file_name, "w+")
        self.write(columns)

    def write(self, values) -> None:
        """
        :param values: list of values. The length of this list needs to be equal to the number
                       of columns
        """
        
        assert len(values) == self.num_columns, f'Not enough values to log! Expected {self.num_columns} values, and got {len(values)}'
        values: List[str] = map(lambda x: x if isinstance(x, str) else f'{x:.4f}', values)
        self.file.write(self.delimiter.join(values) + "\n")
        self.file.flush()