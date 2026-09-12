import json
import re


def parse_llm_json(response):
    """
    Convert Gemma response into a Python dictionary.

    Handles both:
    1. Pure JSON
    2. JSON wrapped inside markdown code fences
    """

    if not response:
        raise ValueError("Empty LLM response")

    response = response.strip()

    # Remove markdown code fences
    response = re.sub(
        r"^```json\s*",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"^```\s*",
        "",
        response
    )

    response = re.sub(
        r"\s*```$",
        "",
        response
    )

    response = response.strip()

    try:
        data = json.loads(response)

    except json.JSONDecodeError as e:

        # Try extracting JSON object
        match = re.search(
            r"\{.*\}",
            response,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                f"Could not find valid JSON in LLM response: {e}"
            )

        data = json.loads(match.group(0))

    # Validate required fields
    required_fields = [
        "defect",
        "confidence",
        "severity",
        "repairability",
        "recommended_action",
        "repair_procedure",
        "estimated_repair_time_minutes",
        "estimated_repair_cost_inr",
        "reason"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )

    return data


if __name__ == "__main__":

    test_response = """
    ```json
    {
        "defect": "Solder Bridge",
        "confidence": 0.91,
        "severity": "Medium",
        "repairability": "High",
        "recommended_action": "REPAIR",
        "repair_procedure": [
            "Inspect the shorted pins or pads.",
            "Remove excess solder.",
            "Clean the area."
        ],
        "estimated_repair_time_minutes": "5 to 20 minutes",
        "estimated_repair_cost_inr": "100 to 250 INR",
        "reason": "High repairability defect."
    }
    ```
    """

    result = parse_llm_json(test_response)

    print("=" * 70)
    print("JSON PARSER TEST")
    print("=" * 70)

    print("\n✓ JSON successfully parsed\n")

    print("Defect        :", result["defect"])
    print("Confidence    :", result["confidence"])
    print("Severity      :", result["severity"])
    print("Repairability :", result["repairability"])
    print("Action        :", result["recommended_action"])
    print("Time          :", result["estimated_repair_time_minutes"])
    print("Cost          :", result["estimated_repair_cost_inr"])

    print("\n✓ PARSER TEST COMPLETED")