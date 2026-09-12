import torch
from ultralytics import YOLO
DATA_YAML = "vision/data.yaml"
def main():
    print("=" * 70)
    print("AUTONOMOUS PCB INSPECTION")
    print("YOLO11 PCB DEFECT TRAINING")
    print("=" * 70)
    # Check GPU
    if torch.cuda.is_available():
        device = 0
        print("\n? CUDA GPU available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\n? CUDA GPU not available")
        print("Using CPU for training")
    print("\nLoading YOLO11n...")
    model = YOLO("yolo11n.pt")
    print("? YOLO11n loaded")
    print("\nStarting training...")
    model.train(
       data=DATA_YAML,
    epochs=10,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    project="runs/pcb",
    name="yolo11_pcb",
    patience=5,
    save=True,
    val=True,
    cache=False,
    verbose=True
)
    print("\n" + "=" * 70)
    print("? TRAINING COMPLETED")
    print("=" * 70)
    print("\nBest model:")
    print("runs/pcb/yolo11_pcb/weights/best.pt")
if __name__ == "__main__":
    main()
