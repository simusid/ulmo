import os

class ExperimentManager:
    def __init__(self, base_dir='runs', base_name='exp'):
        """
        Initialize the ExperimentManager with a base directory and base name for subdirectories.

        :param base_dir: The base directory to store experiments (default: 'runs').
        :param base_name: The base name for experiment subdirectories (default: 'exp').
        """
        self.base_dir = base_dir
        self.base_name = base_name

    def new(self):
        """
        Create a new experiment directory under the base directory. If the base directory
        or experiment directories don't exist, they will be created.

        :return: The path to the newly created experiment directory.
        """
        # Ensure the base directory exists
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        # Find the next available experiment directory name
        next_index = 1
        while True:
            exp_dir_name = f"{self.base_name}{next_index}"
            exp_dir_path = os.path.join(self.base_dir, exp_dir_name)
            if not os.path.exists(exp_dir_path):
                os.makedirs(exp_dir_path)
                return exp_dir_path
            next_index += 1

# Example usage
if __name__ == "__main__":
    manager = ExperimentManager(base_dir='runs', base_name='exp')
    new_exp_path = manager.new()
    print(f"New experiment directory created: {new_exp_path}")
