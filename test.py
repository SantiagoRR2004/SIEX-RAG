from datasets import load_dataset, concatenate_datasets
import random
import chatbotMITRE


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


if __name__ == "__main__":

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

    chat = testChatbot()
    chat.main()

    print(f"{input}")

    print(f"{output}")
