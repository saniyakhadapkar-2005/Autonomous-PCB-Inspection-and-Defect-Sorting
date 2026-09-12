# from langchain_ollama import OllamaLLM


# MODEL_NAME = "gemma3:4b-it-qat"


# def get_llm():
#     """
#     Load Gemma 3 4B locally through Ollama.
#     """

#     return OllamaLLM(
#         model=MODEL_NAME,
#         temperature=0.1
#     )


# def generate_diagnosis(defect_name, confidence, retrieved_knowledge):
#     """
#     Generate structured PCB defect diagnosis using
#     retrieved RAG knowledge + Gemma 3 4B.
#     """

#     llm = get_llm()

#     knowledge_text = "\n\n".join(
#         item["content"]
#         for item in retrieved_knowledge
#     )

#     prompt = f"""
# You are an expert PCB inspection and repair assistant.

# A YOLO11 computer vision model detected the following PCB defect:

# Defect: {defect_name}
# Detection Confidence: {confidence}

# Relevant PCB repair knowledge retrieved from the knowledge base:

# ---------------- KNOWLEDGE ----------------
# {knowledge_text}
# --------------------------------------------

# Based ONLY on the detected defect and the retrieved knowledge,
# analyze the PCB defect.

# Return the answer ONLY as valid JSON.

# Use exactly this structure:

# {{
#     "defect": "{defect_name}",
#     "confidence": {confidence},
#     "severity": "Low / Medium / High / Critical",
#     "repairability": "Low / Medium / High",
#     "recommended_action": "REPAIR / REJECT",
#     "repair_procedure": [
#         "Step 1",
#         "Step 2",
#         "Step 3"
#     ],
#     "estimated_repair_time_minutes": "value or range",
#     "estimated_repair_cost_inr": "value or range",
#     "reason": "short explanation"
# }}

# Important rules:

# 1. Do not invent a completely different defect.
# 2. Use the retrieved knowledge as the primary source.
# 3. Keep cost and time as prototype estimates.
# 4. Do not claim real industrial pricing.
# 5. Do not return markdown.
# 6. Return JSON only.
# """

#     print("\nSending diagnosis request to Gemma 3 4B...")

#     response = llm.invoke(prompt)

#     return response


# if __name__ == "__main__":

#     from rag.retriever import retrieve_relevant_knowledge

#     print("=" * 70)
#     print("GEMMA 3 4B PCB DIAGNOSIS TEST")
#     print("=" * 70)

#     defect = "Solder Bridge"
#     confidence = 0.91

#     print(f"\nDetected Defect : {defect}")
#     print(f"Confidence      : {confidence}")

#     print("\nRetrieving PCB repair knowledge...")

#     knowledge = retrieve_relevant_knowledge(
#         f"PCB repair procedure for {defect}",
#         k=2
#     )

#     print(f"✓ Retrieved {len(knowledge)} knowledge chunks")

#     result = generate_diagnosis(
#         defect_name=defect,
#         confidence=confidence,
#         retrieved_knowledge=knowledge
#     )

#     print("\n" + "=" * 70)
#     print("GEMMA 3 4B DIAGNOSIS")
#     print("=" * 70)

#     print(result)

#     print("\n" + "=" * 70)
#     print("✓ GEMMA DIAGNOSIS COMPLETED")
#     print("=" * 70)



import ollama
from langchain_ollama import OllamaLLM


# ============================================================
# MODEL CONFIGURATION
# ============================================================

PRIMARY_MODEL =  "gemma:2b"

# Lightweight fallback models tried in order if the primary
# model fails to allocate buffers (OOM) or is unavailable.
FALLBACK_MODELS = [
    
]

ALL_MODELS = [PRIMARY_MODEL] + FALLBACK_MODELS


def get_llm(model_name=PRIMARY_MODEL):
    """
    Load a Gemma model locally through Ollama with memory optimizations.

    Lightweight fallback models use a smaller context window to
    further reduce memory pressure.
    """
    num_ctx = 1024 if model_name in FALLBACK_MODELS else 1048

    return OllamaLLM(
        model=model_name,
        temperature=0.1,
        num_ctx=num_ctx,
        num_predict=256,
        format="json"
    )


def _build_prompt(defect_name, confidence, retrieved_knowledge):
    """
    Build the structured JSON diagnosis prompt.
    """
    knowledge_text = "\n\n".join(
        item["content"]
        for item in retrieved_knowledge
    )

    prompt = f"""
You are an expert PCB inspection and repair assistant.

A YOLO11 computer vision model detected the following PCB defect:

Defect: {defect_name}
Detection Confidence: {confidence}

Relevant PCB repair knowledge retrieved from the knowledge base:

---------------- KNOWLEDGE ----------------
{knowledge_text}
--------------------------------------------

Based ONLY on the detected defect and the retrieved knowledge,
analyze the PCB defect.

Return the answer ONLY as valid JSON.

Use exactly this structure:

{{
    "defect": "{defect_name}",
    "confidence": {confidence},
    "severity": "Low / Medium / High / Critical",
    "repairability": "Low / Medium / High",
    "recommended_action": "REPAIR / REJECT",
    "repair_procedure": [
        "Step 1",
        "Step 2",
        "Step 3"
    ],
    "estimated_repair_time_minutes": "value or range",
    "estimated_repair_cost_inr": "value or range",
    "reason": "short explanation"
}}

Important rules:

1. Do not invent a completely different defect.
2. Use the retrieved knowledge as the primary source.
3. Keep cost and time as prototype estimates.
4. Do not claim real industrial pricing.
5. Do not return markdown.
6. Return JSON only.
"""

    return prompt


def generate_diagnosis(defect_name, confidence, retrieved_knowledge):
    """
    Generate structured PCB defect diagnosis using
    retrieved RAG knowledge + Gemma 3 4B.

    If the primary model fails to allocate buffers (OOM) or the
    Ollama server reports an error, this function automatically
    falls back to lightweight models such as gemma:2b.

    Returns:
        dict: {
            "response": str,          # raw LLM response (JSON text)
            "model_used": str,        # model that generated the response
            "fallback_used": bool,    # True if a fallback model was used
            "warning": str | None     # actionable warning if fallback was used
        }

    Raises:
        ollama.ResponseError: If every configured model fails.
        ollama.RequestError:  If the Ollama server is unreachable.
    """

    prompt = _build_prompt(
        defect_name=defect_name,
        confidence=confidence,
        retrieved_knowledge=retrieved_knowledge
    )

    last_error = None

    for index, model_name in enumerate(ALL_MODELS):

        try:

            print(f"\nSending diagnosis request to {model_name}...")

            llm = get_llm(model_name)

            response = llm.invoke(prompt)

            fallback_used = index > 0

            warning = None

            if fallback_used and last_error is not None:

                warning = (
                    f"⚠️ Primary model '{PRIMARY_MODEL}' failed: "
                    f"{last_error.error}. Diagnosis was generated using "
                    f"the lightweight fallback model '{model_name}'. "
                    f"To restore full accuracy, free GPU/CPU memory, "
                    f"run 'ollama pull {PRIMARY_MODEL}', and retry."
                )

            return {
                "response": response,
                "model_used": model_name,
                "fallback_used": fallback_used,
                "warning": warning
            }

        except ollama.ResponseError as e:

            last_error = e

            print(
                f"✗ {model_name} failed: {e.error} "
                f"(status code: {e.status_code})"
            )

        except ollama.RequestError as e:

            # Ollama server is unreachable — all models will fail.
            print(f"✗ Ollama server unreachable: {e.error}")
            raise

    # Every configured model failed — surface the last error.
    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "No Ollama model could be used to generate a diagnosis."
    )


if __name__ == "__main__":

    from rag.retriever import retrieve_relevant_knowledge

    print("=" * 70)
    print("GEMMA 3 4B PCB DIAGNOSIS TEST")
    print("=" * 70)

    defect = "Solder Bridge"
    confidence = 0.91

    print(f"\nDetected Defect : {defect}")
    print(f"Confidence      : {confidence}")

    print("\nRetrieving PCB repair knowledge...")

    knowledge = retrieve_relevant_knowledge(
        f"PCB repair procedure for {defect}",
        k=2
    )

    print(f"✓ Retrieved {len(knowledge)} knowledge chunks")

    result = generate_diagnosis(
        defect_name=defect,
        confidence=confidence,
        retrieved_knowledge=knowledge
    )

    print("\n" + "=" * 70)
    print("GEMMA 3 4B DIAGNOSIS")
    print("=" * 70)

    print(f"Model used : {result['model_used']}")
    print(f"Fallback   : {result['fallback_used']}")

    if result["warning"]:
        print(f"Warning    : {result['warning']}")

    print("\nResponse:")
    print(result["response"])

    print("\n" + "=" * 70)
    print("✓ GEMMA DIAGNOSIS COMPLETED")
    print("=" * 70)
