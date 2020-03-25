from .base import BasePart


class ConsolePrinter(BasePart):
    def __init__(self, input_names, print_length=120):
        self.input_names = input_names
        self.num_inputs = len(input_names)
        self.print_length = print_length

    def run(self, *values):
        value_len = len(values)
        assert self.num_inputs == value_len, \
            f"Expected {self.num_inputs} values, but got {values} with {value_len} values"
        temp_dict = dict(zip(self.input_names, values))
        print(f"{str(temp_dict):<{self.print_length}}", flush=True, end="\r")

    def shutdown(self):
        print()
