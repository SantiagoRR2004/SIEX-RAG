import downloader
import databaseCreator

if __name__ == "__main__":
    downloader.downloadTechniquesEnterpriseAttack()
    databaseCreator.createVectorStore()
