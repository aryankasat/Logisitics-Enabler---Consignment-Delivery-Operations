from llama_stack_client import LlamaStackClient, Agent
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from rich.pretty import pprint
import json
import uuid
from pydantic import BaseModel
import rich
import os
import pickle
from rag import rag

#fw_3ZGTegTnhTMhgi7fLR6FVTFk
os.environ['FIREWORKS_API_KEY'] = "FIREWORKS_API_KEY"
client = LlamaStackAsLibraryClient("fireworks", provider_data = {"fireworks_api_key": os.environ['FIREWORKS_API_KEY']})
_ = client.initialize()
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
company = "Tata Consultancy Services"

base_agent_config = dict(
    model=MODEL_ID,
    instructions="You are an assistant.",
    # sampling_params={
    #     "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
    # },
)
 
# with open("vector_db_tcs_annual_report.pkl", "rb") as f:
#     vector_db = pickle.load(f)
# vector_db_id = vector_db.identifier


def export_guide_rag_agent():
    export_guide_rag_agent =     {
                                        **base_agent_config,
                                        "instructions":'''You are a RAG-enabled assistant. Based on the query asked by the user fetch the information from the document provided. 
                                                                    Do not rely on prior knowledge or generate any content not grounded in the retrieved documents. **Avoid Hallucinations** and 
                                                                    ensure all  outputs are traceable to the source content.''',
                                        # "tools":[{
                                        #     "name": "builtin::rag",
                                        #     "args": {"vector_db_ids": [vector_db_id]},
                                        # }],
    }  
    return Agent(client, **export_guide_rag_agent)

def web_search_agent():
    web_search_agent =  {
                                                    **base_agent_config,
                                                    "instructions":f'''Use the web search tool to retrieve the upcoming potential hurdles that can be caused due to weather or political threats  
                                                                                that come up in the web news which might affect the delivery of the consignment to the location as mentioned in the data.
                                                                                Frame the output correctly with correct headings for further agents to connect it properly.
                                                                                {{
                                                                                    [heading] : [text data (string)],
                                                                                }}
                                                                                ''',
                                                    "tools":["builtin::websearch"]
    }

    return Agent(client, **web_search_agent)

def planning_and_dispatch_agent():
    planning_and_dispatch_agent = {
                                                        **base_agent_config,
                                                        "instructions": '''Analyze the data retrieved from the **rag_agent**, and generate actionable insights and 
                                                                                recommended planning and dispatching strategy. Finally compile all the data systematically for the end user.
                                                                                The format of the data should be as follows -
                                                                                {{
                                                                                    [sub-heading-1] : [text data (string)],
                                                                                    [sub-heading-2] : [text data (string)],
                                                                                }}
                                                                                '''
    }
    return Agent(client, **planning_and_dispatch_agent)

def delivery_pipeline_forecasting_agent():
    delivery_pipeline_forecasting_agent ={

                                                    **base_agent_config,
                                                    "instructions": """Based on the data received from the web agent regarding potential hurdles and query received from the user suggest a 
                                                                                suitable plan and come up with a time frame + buffer for delivery of the consignment.
                                                                                The format of the data returned should be as follows - 
                                                                                {{
                                                                                    [heading]:[pipeline]
                                                                                }}
                                                                                """
    }
    return Agent(client, **delivery_pipeline_forecasting_agent)
                                                    
def orchestration_agent():
    orchestration_agent = {      
                                                    **base_agent_config,
                                                    "instructions": '''Intelligently organize and synthesize all incoming data to provide meaningful and actionable insights for the end user.
                                                                              ensuring proper delivery of consignment.''',
    }
    return Agent(client, **orchestration_agent)

specialized_agents = {
    "export_guide_rag_agent":export_guide_rag_agent(),
    "web_search_agent":web_search_agent(),
    "planning_and_dispatch_agent":planning_and_dispatch_agent(),
    "delivery_pipeline_forecasting_agent":delivery_pipeline_forecasting_agent(),
    "orchestration_agent":orchestration_agent()
}


