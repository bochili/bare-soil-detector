from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolo11m.pt")
    results = model.train(data="soil.yaml", epochs=100, device="mps") # For Apple Silicon training
    # results = model.train(data="soil.yaml", epochs=100) # For CUDA/CPU training
    model.val()