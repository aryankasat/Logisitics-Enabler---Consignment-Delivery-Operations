# importing dependency
from agents import export_guide_rag_agent, web_search_agent, delivery_pipeline_forecasting_agent, planning_and_dispatch_agent, orchestration_agent
import rich
import uuid
import json
from llama_stack_client.types import Document
import streamlit as st


# Initialize agents
specialized_agents = {
    "export_guide_rag_agent": export_guide_rag_agent(),
    "web_search_agent": web_search_agent(),
    "planning_and_dispatch_agent": planning_and_dispatch_agent(),
    "delivery_pipeline_forecasting_agent": delivery_pipeline_forecasting_agent(),
    "orchestration_agent": orchestration_agent()
}

# Session IDs
specialized_agents_session_ids = {
    name + "_session_id": agent.create_session(session_name=f"{name}_{uuid.uuid4()}")
    for name, agent in specialized_agents.items()
}

# Static document used in the pipeline
document = [
    Document(
        document_id="Goods Export Guide",
        content="export_documentation_guide.pdf",
        mime_type="text/pdf",
        metadata={},
    )
]

# Agent pipeline logic
def process_user_query(user_query):
    try:
        rich.print("[bold magenta]⚙️ Starting Multi-Agent Agentic AI pipeline...[/bold magenta]")

        rich.print(f"🔀 [cyan] Routing Result: Extracting data from the export guide for the delivery of the consignment.  [/cyan]")
        rich.print(f"🔀 [cyan] Routing to RAG Agent... [/cyan]")
        rag_response = specialized_agents["export_guide_rag_agent"].create_turn(
            messages=[{"role": "user", "content": user_query}],
            session_id=specialized_agents_session_ids["export_guide_rag_agent_session_id"],
            documents=document,
            stream=False,
        )
        rag_result = rag_response.output_message.content
        print(rag_result, "\n")

        rich.print(f"🔀 [cyan] Routing Result: Gathering insights about the potential roadblocks like weather hazards or construction work from the web for the delivery of the consignment. [/cyan]")
        rich.print(f"🔀 [cyan] Routing to Web Search Agent...[/cyan]")
        website_response = specialized_agents["web_search_agent"].create_turn(
            messages=[{"role": "user", "content": user_query}],
            session_id=specialized_agents_session_ids["web_search_agent_session_id"],
            stream=False,
        )
        website_result = website_response.output_message.content
        print(website_result, "\n")

        rich.print(f"🔀 [cyan] Routing Result: Based on the export guide documents list, planning for the dispatch of the consignment.[/cyan]")
        rich.print(f"🔀 [cyan] Routing to Planning and Dispatch Agent...[/cyan]")
        planning_dispatch_response = specialized_agents["planning_and_dispatch_agent"].create_turn(
            messages=[{"role": "user", "content": rag_result + "\n" + website_result}],
            session_id=specialized_agents_session_ids["planning_and_dispatch_agent_session_id"],
            stream=False,
        )
        planning_dispatch_result = planning_dispatch_response.output_message.content
        print(planning_dispatch_result, "\n")

        rich.print(f"🔀 [cyan] Routing Result: Analyzing the relevant data to forecaste the delivery plan for consignment..[/cyan]")
        rich.print(f"🔀 [cyan] Routing to Delivery Forecasting Agent...[/cyan]")
        delivery_forecast_response = specialized_agents["delivery_pipeline_forecasting_agent"].create_turn(
            messages=[{"role": "user", "content": rag_result + "\n" + website_result}],
            session_id=specialized_agents_session_ids["delivery_pipeline_forecasting_agent_session_id"],
            stream=False,
        )
        delivery_forecast_result = delivery_forecast_response.output_message.content
        print(delivery_forecast_result, "\n")

        rich.print(f"🔀 [cyan] Routing Result: Orchestrating the data to be consumed by the end user.[/cyan]")
        rich.print(f"🔀 [cyan] Routing to Orchestration Agent...[/cyan]")
        orchestration_response = specialized_agents["orchestration_agent"].create_turn(
            messages=[{"role": "user", "content": planning_dispatch_result + "\n" + delivery_forecast_result }],
            session_id=specialized_agents_session_ids["orchestration_agent_session_id"],
            stream=False,
        )
        orchestration_result = orchestration_response.output_message.content
        print(orchestration_result, "\n")

        return orchestration_result

    except json.JSONDecodeError:
        print("Error: Invalid JSON response from agent.")
        return None
    

# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Logisitics Enabler - Consignment Delivery Operations")

    user_query = st.text_area("User Query", value='''I have a consignment of mangoes to be delivered to the united states of america in the next week. 
                                                                                    Devise for the a complete guide of consignment delivery such that I dont face any problem during the delivery.''')
    

    if st.button("Run Analysis"):
        result = process_user_query(user_query)
        st.subheader("🧩 Final Consignment Delivery Plan")
        st.write(result)
    

if __name__ == "__main__":
    main()
