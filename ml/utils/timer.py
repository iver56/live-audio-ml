import time


class timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s
    """

    def __init__(self, description="Execution time"):
        self.description = description

    def __enter__(self):
        self.t = time.clock()
        return self

    def __exit__(self, type, value, traceback):
        print("{}: {:.3f} s".format(self.description, time.clock() - self.t))


if __name__ == "__main__":
    with timer("A simple test") as t:
        time.sleep(1.337)
