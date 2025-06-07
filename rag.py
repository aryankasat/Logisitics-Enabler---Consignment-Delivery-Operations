from llama_stack_client.types import Document
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
import os
import pickle

os.environ['FIREWORKS_API_KEY'] = "fw_3ZGTegTnhTMhgi7fLR6FVTFk"
client = LlamaStackAsLibraryClient("fireworks", provider_data = {"fireworks_api_key": os.environ['FIREWORKS_API_KEY']})
_ = client.initialize()

   
def rag ():   
        document  =[ Document(
                    document_id="TCS_annual_report",
                    content = "annual-report-2023-2024.pdf",
                    mime_type="application/pdf",
                    metadata={},
                ) 
        ]

        vector_providers = [
            provider for provider in client.providers.list() if provider.api == "vector_io"
        ]
        selected_vector_provider = vector_providers[0]
        vector_db_id = "vector_db_tcs_annual_report"
        vector_db = client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id=selected_vector_provider.provider_id,
        )

        client.tool_runtime.rag_tool.insert(
            documents=document,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
        )

        # with open("vector_db_tcs_annual_report.pkl", "wb") as f:
        #     pickle.dump(vector_db, f)

        return "vector_db_tcs_annual_report"

        


