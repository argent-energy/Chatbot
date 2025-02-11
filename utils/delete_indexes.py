import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
import time

load_dotenv(".env.gpt4", override=True)

# Replace with your search service endpoint and admin key
endpoint = os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"]
admin_key = os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"]

# Create a SearchServiceClient
credential = AzureKeyCredential(admin_key)
client = SearchIndexClient(endpoint=endpoint, credential=credential)

# Get all indexes
indexes = client.list_indexes()

print(f"Currently we have {indexes.__sizeof__()} indexes")
print("Deleting the indexes...")
# Print and delete the index names
for index in indexes:
    # print(index)
    if index.name.startswith("wdcindex") :
        print(f"   [Skipping] {index.name}")
    else :
        print(f"   [Deleted]  {index.name}")
        client.delete_index(index.name)
        time.sleep(5)
        