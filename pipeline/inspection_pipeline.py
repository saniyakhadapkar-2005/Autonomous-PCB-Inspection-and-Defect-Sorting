from pathlib import Path

from vision.detector import PCBDetector
from rag.retriever import retrieve_relevant_knowledge
from rag.generator import generate_diagnosis
from rag.parser import parse_llm_json
from pipeline.decision_engine import make_decision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"runs\detect\runs\pcb\yolo11_pcb-2\weights\best.pt"


# ============================================================
# INITIALIZE YOLO
# ============================================================

def load_detector():
    """
    Load the trained YOLO11 PCB defect detection model.
    """

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Trained YOLO model not found:\n{MODEL_PATH}"
        )

    print("\nLoading trained YOLO11 model...")

    detector = PCBDetector(
        model_path=MODEL_PATH
    )

    print("✓ YOLO11 detector ready")

    return detector


# ============================================================
# COMPLETE INSPECTION PIPELINE
# ============================================================

def inspect_pcb(image_path, confidence_threshold=0.25):
    """
    Complete PCB inspection pipeline.

    Flow:

    Image
      ↓
    YOLO11
      ↓
    Defect Detection
      ↓
    RAG
      ↓
    Gemma 3 4B
      ↓
    JSON Parser
      ↓
    Decision Engine
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"PCB image not found:\n{image_path}"
        )

    print("\n" + "=" * 80)
    print("AUTONOMOUS PCB INSPECTION PIPELINE")
    print("=" * 80)

    print(f"\nInput Image:")
    print(image_path)

    # --------------------------------------------------------
    # STEP 1 — YOLO
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("STEP 1 — YOLO11 DEFECT DETECTION")
    print("-" * 80)

    detector = load_detector()

    detections = detector.get_detections(
        image_path=image_path,
        confidence=confidence_threshold
    )

    print(f"\n✓ YOLO detection completed")
    print(f"Detected objects: {len(detections)}")

    # --------------------------------------------------------
    # NO DEFECT
    # --------------------------------------------------------

    if not detections:

        print("\nNo trained PCB defect detected.")

        return {
            "image": str(image_path),
            "detections": [],
            "diagnoses": [],
            "final_decision": "NO_DEFECT_DETECTED"
        }

    # --------------------------------------------------------
    # Print detections
    # --------------------------------------------------------

    print("\nDetected Defects:")

    for index, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"{index}. "
            f"{detection['class_name']} "
            f"| Confidence: {detection['confidence']:.2f} "
            f"| BBox: {detection['bbox']}"
        )

    # --------------------------------------------------------
    # STEP 2 — RAG + GEMMA
    # --------------------------------------------------------

    diagnoses = []

    print("\n" + "-" * 80)
    print("STEP 2 — RAG + GEMMA 3 4B DIAGNOSIS")
    print("-" * 80)

    for index, detection in enumerate(
        detections,
        start=1
    ):

        defect_name = detection["class_name"]
        defect_confidence = detection["confidence"]

        print(
            f"\n[{index}/{len(detections)}] "
            f"Processing: {defect_name}"
        )

        # ----------------------------------------------------
        # RAG retrieval
        # ----------------------------------------------------

        print("→ Retrieving PCB repair knowledge...")

        query = (
            f"PCB repair procedure, severity, "
            f"repairability, cost and time for "
            f"{defect_name}"
        )

        knowledge = retrieve_relevant_knowledge(
            query=query,
            k=2
        )

        print(
            f"✓ Retrieved {len(knowledge)} knowledge chunks"
        )

        # ----------------------------------------------------
        # Gemma diagnosis
        # ----------------------------------------------------

        print("→ Sending information to Gemma 3 4B...")

        llm_result = generate_diagnosis(
            defect_name=defect_name,
            confidence=defect_confidence,
            retrieved_knowledge=knowledge
        )

        raw_response = llm_result["response"]
        model_used = llm_result["model_used"]
        fallback_used = llm_result["fallback_used"]
        llm_warning = llm_result["warning"]

        print(f"✓ Gemma diagnosis received (model: {model_used})")

        if fallback_used:
            print(f"⚠️ Fallback model used: {model_used}")

        # ----------------------------------------------------
        # JSON parsing
        # ----------------------------------------------------

        print("→ Parsing Gemma JSON...")

        diagnosis = parse_llm_json(
            raw_response
        )

        print("✓ JSON parsed successfully")

        # ----------------------------------------------------
        # Decision Engine
        # ----------------------------------------------------

        print("→ Running Decision Engine...")

        decision = make_decision(
            diagnosis
        )

        print(
            f"✓ Decision: {decision['decision']}"
        )

        # ----------------------------------------------------
        # Combine diagnosis + decision
        # ----------------------------------------------------

        combined_result = {
            "detection": detection,
            "diagnosis": diagnosis,
            "decision": decision,
            "model_used": model_used,
            "fallback_used": fallback_used,
            "llm_warning": llm_warning
        }

        diagnoses.append(
            combined_result
        )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print("\n" + "-" * 80)
    print("STEP 3 — FINAL PCB DECISION")
    print("-" * 80)

    decisions = [
        item["decision"]["decision"]
        for item in diagnoses
    ]

    # If any defect requires rejection,
    # reject the complete PCB.

    if "REJECT" in decisions:

        final_decision = "REJECT"

    else:

        final_decision = "REPAIR"

    print(
        f"\nFINAL DECISION: {final_decision}"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {
        "image": str(image_path),
        "detections": detections,
        "diagnoses": diagnoses,
        "final_decision": final_decision
    }

    print("\n" + "=" * 80)
    print("✓ COMPLETE PCB INSPECTION PIPELINE FINISHED")
    print("=" * 80)

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("COMPLETE PCB AI PIPELINE TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # CHANGE THIS IMAGE PATH
    # --------------------------------------------------------

    TEST_IMAGE = r"data\test_images\pcb_test.jpg"

    try:

        result = inspect_pcb(
            image_path=TEST_IMAGE,
            confidence_threshold=0.25
        )

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        print(
            f"\nImage: {result['image']}"
        )

        print(
            f"Detections: "
            f"{len(result['detections'])}"
        )

        print(
            f"Final Decision: "
            f"{result['final_decision']}"
        )

        for index, item in enumerate(
            result["diagnoses"],
            start=1
        ):

            detection = item["detection"]
            diagnosis = item["diagnosis"]
            decision = item["decision"]

            print("\n" + "-" * 60)

            print(
                f"DEFECT {index}: "
                f"{detection['class_name']}"
            )

            print(
                f"Confidence: "
                f"{detection['confidence']}"
            )

            print(
                f"Severity: "
                f"{diagnosis['severity']}"
            )

            print(
                f"Repairability: "
                f"{diagnosis['repairability']}"
            )

            print(
                f"Estimated Time: "
                f"{diagnosis['estimated_repair_time_minutes']}"
            )

            print(
                f"Estimated Cost: "
                f"{diagnosis['estimated_repair_cost_inr']}"
            )

            print(
                f"Decision: "
                f"{decision['decision']}"
            )

            print(
                f"Reason: "
                f"{decision['reason']}"
            )

    except Exception as e:

        print("\n❌ PIPELINE ERROR")
        print(str(e))