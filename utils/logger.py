import sys

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_error(*args, **kwargs):
    msg = " ".join(map(str, args))
    print(f"{Colors.RED}{msg}{Colors.RESET}", **kwargs)

def print_success(*args, **kwargs):
    msg = " ".join(map(str, args))
    print(f"{Colors.GREEN}{msg}{Colors.RESET}", **kwargs)

def print_info(*args, **kwargs):
    msg = " ".join(map(str, args))
    print(f"{Colors.BLUE}{msg}{Colors.RESET}", **kwargs)
