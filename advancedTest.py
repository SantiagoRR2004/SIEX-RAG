import test
from datasets import load_dataset, concatenate_datasets
import io
from contextlib import redirect_stdout
import tqdm


if __name__ == "__main__":
    # Create an in-memory file object to capture output, effectively suppressing it
    suppress_output = io.StringIO()

    # Load the dataset
    dataset = load_dataset("dattaraj/security-attacks-MITRE")

    # We get all the data together
    data = concatenate_datasets([dataset["train"], dataset["validation"]])

    # We set the function that gets the input
    setattr(test.testChatbot, "getUserInput", test.oneTimeUserInput)

    corretTechniques = 0
    total = 0

    for d in tqdm.tqdm(data):
        # We get the input and output of the example
        test.input, output = test.getInputAndOutput(d)
        test.continueChat = True

        tecnique = test.getMITRETechnique(output)

        if tecnique:

            total += 1

            with redirect_stdout(suppress_output):
                chat = test.testChatbot()
                chat.main()

            modelTecnique = test.getMITRETechnique(chat.lastResponse.content)

            if modelTecnique == tecnique:
                corretTechniques += 1

    print(f"Correct techniques: {corretTechniques}/{total}")
