# agentic_workflow/agents/agent2_rag_retriever.py
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agentic_workflow.state import PipelineState
from workflow_config import (
    OPENROUTER_KEY, OPENROUTER_BASE, MODEL_GENERATOR,
    DB_BSDD, DB_EXPRESS,
    RAG_K_BSDD, RAG_K_EXPRESS,
    RAG_FETCH_K_BSDD, RAG_FETCH_K_EXPRESS,
    RAG_LAMBDA_BSDD, RAG_LAMBDA_EXPRESS
)

SYNTHESIS_PROMPT = """You are an IFC schema expert. You have retrieved chunks from two
knowledge bases in response to a building regulation rule.

CRITICAL GROUNDING RULE — read this before doing anything else:
You may ONLY use IFC class names, Pset names, property names, and relationship entity
names that appear VERBATIM in the chunks provided below. Do NOT invent, generalise, or
recall names from training data. If a required name is not present in the chunks, write
"NOT FOUND IN CHUNKS" in its place rather than inventing a name.

Knowledge bases:
1. bSDD (buildingSMART Data Dictionary): element class definitions and property sets (Psets)
2. IFC EXPRESS schema: IFC class hierarchy, relationship entities, attribute definitions

BUILDING RULE: {rule_text}

ELEMENTS TO CHECK: {elements}

PROPERTIES TO CHECK: {properties}

RAW CHUNKS FROM bSDD:
{bsdd_chunks}

RAW CHUNKS FROM IFC EXPRESS SCHEMA:
{express_chunks}

Extract and compile the following sections using ONLY names found verbatim in the chunks:

1. RELEVANT IFC CLASSES
   For each building element in the rule: copy the exact IFC class name from the chunks
   and add a one-line explanation. If not found, write "NOT FOUND IN CHUNKS".

2. RELEVANT PSETS AND PROPERTIES
   For each property to check: copy the exact Pset name and property name from the chunks,
   including the expected data type. If not found, write "NOT FOUND IN CHUNKS".

3. RELEVANT RELATIONSHIPS
   List the IFC relationship entities needed to navigate between elements.
   Copy names verbatim from the chunks. If not found, write "NOT FOUND IN CHUNKS".

4. IFCOPENSHELL ACCESS PATTERN
   Describe how to access each element and property using ifcopenshell,
   based strictly on what the chunks show.

Be concise. Omit anything not directly relevant to the rule."""


# --- GROUNDING CHECK ---

_IFC_NAME_PATTERN = re.compile(r'\bIfc[A-Z][a-zA-Z0-9]+\b')
_PSET_NAME_PATTERN = re.compile(r'\b(?:Pset|Qto)_[A-Za-z0-9]+\b')


def _check_grounding(synthesis: str, chunks_bsdd: list, chunks_express: list) -> tuple:
    """
    Verify that IFC entity names and Pset/Qto names in the synthesis
    appear verbatim in the raw chunks.
    Returns (annotated_synthesis, list_of_unverified_names).
    """
    all_chunks = " ".join(chunks_bsdd + chunks_express)

    names_in_synthesis = (
        set(_IFC_NAME_PATTERN.findall(synthesis))
        | set(_PSET_NAME_PATTERN.findall(synthesis))
    )

    unverified = sorted(
        name for name in names_in_synthesis
        if name not in all_chunks
    )

    if unverified:
        warning = (
            "\n\n⚠ GROUNDING WARNING — The following names appear in this summary "
            "but could NOT be verified in the retrieved chunks. "
            "They may be hallucinated. Treat with caution:\n"
            + ", ".join(unverified)
        )
        return synthesis + warning, unverified

    return synthesis, []


def _setup_retrievers():
    """Initialise both vector database retrievers using MMR for diversity."""
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=OPENROUTER_KEY,
        openai_api_base=OPENROUTER_BASE
    )

    retriever_bsdd = Chroma(
        persist_directory=DB_BSDD,
        embedding_function=embeddings
    ).as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RAG_K_BSDD,
            "fetch_k": RAG_FETCH_K_BSDD,
            "lambda_mult": RAG_LAMBDA_BSDD
        }
    )

    retriever_express = Chroma(
        persist_directory=DB_EXPRESS,
        embedding_function=embeddings
    ).as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RAG_K_EXPRESS,
            "fetch_k": RAG_FETCH_K_EXPRESS,
            "lambda_mult": RAG_LAMBDA_EXPRESS
        }
    )

    return retriever_bsdd, retriever_express


def _retrieve_all_queries(
    retriever_bsdd, retriever_express,
    queries_bsdd: list, queries_express: list
) -> tuple:
    """
    Retrieve chunks from each database using database-specific queries.
    Deduplicates results across queries.
    """
    seen_bsdd = set()
    seen_express = set()
    chunks_bsdd = []
    chunks_express = []

    for query in queries_bsdd:
        for doc in retriever_bsdd.invoke(query):
            if doc.page_content not in seen_bsdd:
                seen_bsdd.add(doc.page_content)
                chunks_bsdd.append(doc.page_content)

    for query in queries_express:
        for doc in retriever_express.invoke(query):
            if doc.page_content not in seen_express:
                seen_express.add(doc.page_content)
                chunks_express.append(doc.page_content)

    return chunks_bsdd, chunks_express


def run_agent_2(state: PipelineState) -> PipelineState:
    """
    Agent 2: RAG Retriever
    Input:  state.parsed_rule (search_queries_bsdd, search_queries_express, primary_element, related_elements, properties_to_check)
    Output: state.rag_context (synthesised IFC knowledge for Agent 3)
    """

    print("\n[Agent 2] Starting RAG retrieval...")

    parsed = state.parsed_rule
    all_elements = [parsed.primary_element] + parsed.related_elements
    queries_bsdd = parsed.search_queries_bsdd
    queries_express = parsed.search_queries_express

    print(f"[Agent 2] bSDD queries:    {queries_bsdd}")
    print(f"[Agent 2] EXPRESS queries: {queries_express}")

    try:
        # --- Step 1: Retrieve chunks from both databases ---
        retriever_bsdd, retriever_express = _setup_retrievers()
        chunks_bsdd, chunks_express = _retrieve_all_queries(
            retriever_bsdd, retriever_express, queries_bsdd, queries_express
        )

        print(f"[Agent 2] Retrieved {len(chunks_bsdd)} bSDD chunks, "
              f"{len(chunks_express)} EXPRESS chunks (deduplicated)")

        # --- Step 2: Synthesise chunks into structured context ---
        llm = ChatOpenAI(
            model_name=MODEL_GENERATOR,
            openai_api_key=OPENROUTER_KEY,
            openai_api_base=OPENROUTER_BASE,
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | llm | StrOutputParser()

        rag_context = chain.invoke({
            "rule_text": state.building_rule_text,
            "elements": ", ".join(all_elements),
            "properties": ", ".join(parsed.properties_to_check),
            "bsdd_chunks": "\n---\n".join(chunks_bsdd) if chunks_bsdd else "No relevant chunks found.",
            "express_chunks": "\n---\n".join(chunks_express) if chunks_express else "No relevant chunks found."
        })

        # --- Step 3: Grounding check ---
        rag_context, unverified = _check_grounding(rag_context, chunks_bsdd, chunks_express)

        if unverified:
            print(f"[Agent 2] ⚠ Grounding warning — {len(unverified)} unverified name(s): "
                  f"{', '.join(unverified)}")
        else:
            print("[Agent 2] Grounding check passed — all names verified in chunks.")

        print("[Agent 2] RAG context synthesised successfully.")

        return state.model_copy(update={
            "rag_context": rag_context,
            "retrieved_chunks_bsdd": chunks_bsdd,
            "retrieved_chunks_express": chunks_express,
            "current_agent": "agent_3"
        })

    except Exception as e:
        print(f"[Agent 2] ERROR: {e}")
        return state.model_copy(update={
            "error_log": state.error_log + [f"Agent 2 failed: {str(e)}"],
            "pipeline_failed": True,
            "current_agent": "error"
        })
