from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
print(next(model.model.parameters()).device)
results = model("/home/omid/Age-Estimation/data/raw/White-shark-3.jpg", device="cuda")
print(next(model.model.parameters()).device)
print(results)