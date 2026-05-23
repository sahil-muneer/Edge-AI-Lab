from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- SHARED DATA STRUCTURES ---
class Point(BaseModel):
    x: float
    y: float

class PointWithClass(BaseModel):
    x: float
    y: float
    class_id: int

class RegressionRequest(BaseModel):
    points: List[Point]

class PolyRequest(BaseModel):
    points: List[Point]
    degree: int

class KNNBoundaryRequest(BaseModel):
    points: List[PointWithClass]
    k: int

# --- 1. LINEAR REGRESSION ---
@app.post("/api/linear-regression")
async def calculate_regression(req: RegressionRequest):
    if len(req.points) < 2:
        return {"slope": 0, "intercept": 0}
    X = np.array([[p.x] for p in req.points])
    y = np.array([p.y for p in req.points])
    model = LinearRegression().fit(X, y)
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_)}

# --- 2. POLYNOMIAL REGRESSION (OVERFITTING SANDBOX) ---
@app.post("/api/polynomial-regression")
async def calculate_poly(req: PolyRequest):
    if len(req.points) < 2:
        return {"curve": []}
    
    X = np.array([[p.x] for p in req.points])
    y = np.array([p.y for p in req.points])
    
    # Create polynomial features (this bends the line)
    poly = PolynomialFeatures(degree=req.degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    # Generate 100 points across the screen to draw the smooth curved path
    X_plot = np.linspace(0, 1000, 100).reshape(-1, 1)
    y_plot = model.predict(poly.transform(X_plot))
    
    curve = [{"x": float(x[0]), "y": float(y)} for x, y in zip(X_plot, y_plot)]
    return {"curve": curve}

# --- 3. KNN & DECISION BOUNDARIES ---
@app.post("/api/knn-boundary")
async def calculate_knn_boundary(req: KNNBoundaryRequest):
    if len(req.points) == 0:
        return {"grid": []}

    X_train = np.array([[p.x, p.y] for p in req.points])
    y_train = np.array([p.class_id for p in req.points])

    # Ensure K isn't larger than the dataset
    safe_k = min(req.k, len(X_train))
    model = KNeighborsClassifier(n_neighbors=safe_k).fit(X_train, y_train)

    # Create a 40x40 pixel grid across the screen to generate a Heatmap
    grid_points = []
    for x in range(0, 1000, 40):
        for y in range(0, 600, 40):
            grid_points.append([x, y])
            
    predictions = model.predict(grid_points)
    
    result = []
    for pt, pred in zip(grid_points, predictions):
        result.append({"x": pt[0], "y": pt[1], "class_id": int(pred)})
        
    return {"grid": result}