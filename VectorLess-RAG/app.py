import os
import json
import time
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pageindex import PageIndexClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

api_key = os.getenv("PAGEINDEX_API_KEY")
pi_client = PageIndexClient(api_key=api_key)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

result = pi_client.submit_document("./MERN-report.pdf")
doc_id = result["doc_id"]

while True:
    status_result = pi_client.get_document(doc_id)
    status = status_result.get("status", "pending")
    if status in ["completed", "ready", "indexed"]:
        break
    elif status == "failed":
        raise Exception("Document indexing failed.")
    time.sleep(3)

tree_result = pi_client.get_tree(doc_id, node_summary=True)
pageindex_tree = tree_result.get("result", [])

class TreeSearchResponse(BaseModel):
    thinking: str = Field(description="Step-by-step reasoning on why these nodes are relevant.")
    node_list: List[str] = Field(description="List of node IDs that contain information relevant to the query.")

structured_llm = llm.with_structured_output(TreeSearchResponse)

def llm_tree_search(query: str, tree: list) -> TreeSearchResponse:
    def compress(nodes):
        out = []
        for n in nodes:
            entry = {
                "node_id": n["node_id"],
                "title": n["title"],
                "page": n.get("page_index", "?"),
                "summary": (n.get("text") or n.get("summary") or "")[:150]
            }
            if n.get("nodes"):
                entry["children"] = compress(n["nodes"])
            out.append(entry)
        return out

    compressed_tree = compress(tree)
    prompt = f"""You are a helpful assistant analyzing a document structure to find relevant sections.
Review the document tree below and identify which node IDs are highly relevant to answering the user's query.

Query:
{query}

Tree:
{json.dumps(compressed_tree, indent=2)}
"""
    return structured_llm.invoke([HumanMessage(content=prompt)])

def find_nodes_by_ids(tree: list, target_ids: list) -> list:
    found = []
    for node in tree:
        if node["node_id"] in target_ids:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], target_ids))
    return found

def generate_answer(query: str, nodes: list) -> str:
    if not nodes:
        return "⚠️ No relevant sections found in the document."
    
    context_parts = []
    for node in nodes:
        context_parts.append(
            f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
            f"{node.get('text', 'Content not available.')}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are an expert document analyst.
Answer the question using ONLY the provided context.
For every claim you make, cite the section title and page number in parentheses.
Be concise and precise.

Question: {query}

Context:
{context}

Answer:"""
    
    # Switched from OpenAI to your Groq LLM instance
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def vectorless_rag(query: str, tree: list) -> str:
    search_result = llm_tree_search(query, tree)
    node_ids = getattr(search_result, "node_list", [])
    nodes = find_nodes_by_ids(tree, node_ids)
    return generate_answer(query, nodes)

answer = vectorless_rag(
    query="What are the functional requirement of krishik bazar?",
    tree=pageindex_tree
)
print(answer)