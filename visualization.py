import os


def getTerminalSize() -> tuple:
    """
    Get the size of the terminal window in characters.

    Args:
        - None

    Returns:
        - tuple: (width, height) of the terminal window in characters
    """
    try:
        terminalSize = os.get_terminal_size()
        terminalWidth = terminalSize.columns
        terminalHeight = terminalSize.lines
    except OSError:
        # Default dimensions if terminal size cannot be determined
        terminalWidth = 80
        terminalHeight = 24

    return terminalWidth, terminalHeight


def centerText(text: str, *, fillchar: str = "*", nFill: int = 2) -> str:
    """
    Center text in the terminal window. It adds the
    number of fill characters specified by nFill to
    the left and right of the text.

    Args:
        - text (str): Text to center
        - fillchar (str): Character to use for filling
        - nFill (int): Number of fill characters to use

    Returns:
        - str: Text centered in the terminal window
    """
    term = getTerminalSize()
    return fillchar * nFill + text.center(term[0] - 2 * nFill) + fillchar * nFill + "\n"
