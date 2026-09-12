import re


# ============================================================
# CONFIGURATION
# ============================================================

MAX_REPAIR_COST_INR = 1000
MAX_REPAIR_TIME_MINUTES = 60

REPAIRABLE_LEVELS = {
    "High",
    "Medium"
}

REJECT_SEVERITIES = {
    "Critical"
}


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(text):
    """
    Extract numerical values from strings such as:

    '100 to 250 INR'
    '5-20 minutes'
    '30 minutes'
    """

    if not text:
        return []

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        str(text)
    )

    return [float(number) for number in numbers]


def get_max_value(text):
    """
    Return the maximum numerical value.
    """

    numbers = extract_numbers(text)

    if not numbers:
        return 0

    return max(numbers)


# ============================================================
# DECISION ENGINE
# ============================================================

def make_decision(diagnosis):
    """
    Determine whether the PCB should be repaired or rejected.

    This is a deterministic rule-based decision layer.
    """

    defect = diagnosis.get("defect", "Unknown")

    severity = diagnosis.get(
        "severity",
        "Critical"
    ).strip()

    repairability = diagnosis.get(
        "repairability",
        "Low"
    ).strip()

    repair_time = diagnosis.get(
        "estimated_repair_time_minutes",
        "0"
    )

    repair_cost = diagnosis.get(
        "estimated_repair_cost_inr",
        "0"
    )

    llm_action = diagnosis.get(
        "recommended_action",
        "REJECT"
    ).strip().upper()

    max_time = get_max_value(repair_time)
    max_cost = get_max_value(repair_cost)

    reasons = []

    # --------------------------------------------------------
    # RULE 1 — Critical severity
    # --------------------------------------------------------

    if severity in REJECT_SEVERITIES:

        reasons.append(
            "Defect severity is Critical."
        )

        return {
            "decision": "REJECT",
            "defect": defect,
            "severity": severity,
            "repairability": repairability,
            "estimated_time": repair_time,
            "estimated_cost": repair_cost,
            "reason": " ".join(reasons)
        }

    # --------------------------------------------------------
    # RULE 2 — Low repairability
    # --------------------------------------------------------

    if repairability not in REPAIRABLE_LEVELS:

        reasons.append(
            "Repairability is too low."
        )

        return {
            "decision": "REJECT",
            "defect": defect,
            "severity": severity,
            "repairability": repairability,
            "estimated_time": repair_time,
            "estimated_cost": repair_cost,
            "reason": " ".join(reasons)
        }

    # --------------------------------------------------------
    # RULE 3 — Repair time
    # --------------------------------------------------------

    if max_time > MAX_REPAIR_TIME_MINUTES:

        reasons.append(
            f"Estimated repair time exceeds "
            f"{MAX_REPAIR_TIME_MINUTES} minutes."
        )

        return {
            "decision": "REJECT",
            "defect": defect,
            "severity": severity,
            "repairability": repairability,
            "estimated_time": repair_time,
            "estimated_cost": repair_cost,
            "reason": " ".join(reasons)
        }

    # --------------------------------------------------------
    # RULE 4 — Repair cost
    # --------------------------------------------------------

    if max_cost > MAX_REPAIR_COST_INR:

        reasons.append(
            f"Estimated repair cost exceeds "
            f"₹{MAX_REPAIR_COST_INR}."
        )

        return {
            "decision": "REJECT",
            "defect": defect,
            "severity": severity,
            "repairability": repairability,
            "estimated_time": repair_time,
            "estimated_cost": repair_cost,
            "reason": " ".join(reasons)
        }

    # --------------------------------------------------------
    # RULE 5 — LLM recommendation
    # --------------------------------------------------------

    if llm_action == "REJECT":

        reasons.append(
            "Gemma recommended rejection."
        )

        return {
            "decision": "REJECT",
            "defect": defect,
            "severity": severity,
            "repairability": repairability,
            "estimated_time": repair_time,
            "estimated_cost": repair_cost,
            "reason": " ".join(reasons)
        }

    # --------------------------------------------------------
    # ALL CONDITIONS PASSED
    # --------------------------------------------------------

    reasons.append(
        "Defect is considered repairable."
    )

    reasons.append(
        "Severity is within acceptable range."
    )

    reasons.append(
        "Estimated repair time is within the configured limit."
    )

    reasons.append(
        "Estimated repair cost is within the configured limit."
    )

    return {
        "decision": "REPAIR",
        "defect": defect,
        "severity": severity,
        "repairability": repairability,
        "estimated_time": repair_time,
        "estimated_cost": repair_cost,
        "reason": " ".join(reasons)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_diagnosis = {
        "defect": "Solder Bridge",
        "confidence": 0.91,
        "severity": "Medium",
        "repairability": "High",
        "recommended_action": "REPAIR",
        "repair_procedure": [
            "Inspect the shorted pins.",
            "Remove excess solder.",
            "Clean the area."
        ],
        "estimated_repair_time_minutes": "5 to 20 minutes",
        "estimated_repair_cost_inr": "100 to 250 INR",
        "reason": "High repairability defect."
    }

    result = make_decision(test_diagnosis)

    print("=" * 70)
    print("PCB DECISION ENGINE TEST")
    print("=" * 70)

    print("\nDefect        :", result["defect"])
    print("Severity      :", result["severity"])
    print("Repairability :", result["repairability"])
    print("Estimated Time:", result["estimated_time"])
    print("Estimated Cost:", result["estimated_cost"])

    print("\nFINAL DECISION :", result["decision"])

    print("\nReason:")
    print(result["reason"])

    print("\n" + "=" * 70)
    print("✓ DECISION ENGINE TEST COMPLETED")
    print("=" * 70)