import os
import requests


def downloadTechniquesEnterpriseAttack() -> None:
    """
    Download the Enterprise ATT&CK Techniques from the ATT&CK website and save it to a file.

    It first checks if the file already exists, if it does, it does nothing.

    Args:
        - None

    Returns:
        - None
    """
    url = "https://ccia.esei.uvigo.es/docencia/SIEX/2425/practicas/techniques_enterprise_attack.json"

    folder = os.path.dirname(os.path.abspath(__file__))

    folderPath = os.path.join(folder, "data")

    # If the folder does not exist, create it
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)

    fileName = os.path.join(folderPath, "techniquesEnterpriseAttack.json")

    # If the file already exists, do nothing
    if os.path.exists(fileName):
        pass

    # Otherwise, download the file
    else:
        response = requests.get(url)

        with open(fileName, "w") as f:
            f.write(response.text)


if __name__ == "__main__":
    downloadTechniquesEnterpriseAttack()
