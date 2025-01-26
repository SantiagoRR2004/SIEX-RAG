from datasets import load_dataset, concatenate_datasets
import random
import chatbotMITRE
import io
from contextlib import redirect_stdout
import visualization
import os
import re


class testChatbot(chatbotMITRE.MITREATTACKChatbot):
    pass


def getInputAndOutput(example: dict) -> tuple:
    """
    Get the input and output of the example

    Args:
        - example (dict): The example from the dataset

    Returns:
        - tuple: The input and output of the example
    """
    text = example["text"]

    # The input is the text between "<|user|>" and "<|end|>"
    input = text.split("<|end|>")[0].split("<|user|>")[1].strip()

    # The output is the text between "<|assistant|>" and "<|end|>"
    output = text.split("<|end|>")[1].split("<|assistant|>")[1].strip()

    # Remove the b''
    input = input[2:-1]
    output = output[2:-1]

    return input, output


def oneTimeUserInput(self) -> str:
    """
    This is a function so we can use
    the chatbot to test it.

    It requires to have defined
    outside the input variable
    and counter flag.
    """
    global continueChat
    if continueChat:
        continueChat = False
        return input
    else:
        return ":exit"


def getMITRETechnique(text: str) -> str:
    """
    Get the MITRE technique from the text

    Args:
        - text (str): The text

    Returns:
        - str: The MITRE technique
    """
    # Regular expression to extract only the technique identifier
    pattern = r"T\d{4}(\.\d{3})?"

    # We get the MITRE technique
    match = re.search(pattern, text)

    # If we find the technique, we return it
    if match:
        return match.group(0)
    return None


if __name__ == "__main__":
    # Get the terminal size
    terminal = visualization.getTerminalSize()

    # Load the dataset
    dataset = load_dataset("dattaraj/security-attacks-MITRE")

    # We get all the data together
    data = concatenate_datasets([dataset["train"], dataset["validation"]])

    # We get a random example
    example = random.choice(data)

    # We get the input and output of the example
    input, output = getInputAndOutput(example)
    continueChat = True

    # We set the function that gets the input
    setattr(testChatbot, "getUserInput", oneTimeUserInput)

    # Create an in-memory file object to capture output, effectively suppressing it
    suppress_output = io.StringIO()

    with redirect_stdout(suppress_output):
        chat = testChatbot()
        chat.main()

    # Clear the terminal screen
    os.system("cls" if os.name == "nt" else "clear")

    print(" Query given ".center(terminal[0], "="))
    print(input)

    print(" Expected output ".center(terminal[0], "="))
    print(output)

    print(" Model response ".center(terminal[0], "="))
    print(chat.lastResponse.content)
