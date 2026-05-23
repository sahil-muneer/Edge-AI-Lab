/* eslint-disable */
import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

function App() {
  const [activeTab, setActiveTab] = useState('linear')
  const [isEdgeMode, setIsEdgeMode] = useState(false)
  const [simulateNoise, setSimulateNoise] = useState(false) 
  const [showCppModal, setShowCppModal] = useState(false) 
  
  // States for Manual Data Entry
  const [showDataModal, setShowDataModal] = useState(false)
  const [manualX, setManualX] = useState('')
  const [manualY, setManualY] = useState('')

  // 🚀 NEW: Rolling System Activity Telemetry Logs
  const [logs, setLogs] = useState([
    { time: new Date().toLocaleTimeString(), type: 'SYSTEM', msg: 'Edge AI Core Initialized Successfully.' },
    { time: new Date().toLocaleTimeString(), type: 'HARDWARE', msg: 'Target Connection Status: ESP32-CAM Stream Online.' }
  ])

  const svgRef = useRef(null)
  
  // Tab 1: Linear State
  const [linearPoints, setLinearPoints] = useState([])
  const [regressionData, setRegressionData] = useState({ slope: 0, intercept: 0 })

  // Tab 2: KNN State
  const [kValue, setKValue] = useState(3)
  const [knnBoundary, setKnnBoundary] = useState([])
  const baseKnnData = [
    { x: 150, y: 150, class_id: 0 }, { x: 200, y: 100, class_id: 0 }, { x: 100, y: 200, class_id: 0 }, { x: 250, y: 150, class_id: 0 },
    { x: 650, y: 350, class_id: 1 }, { x: 700, y: 300, class_id: 1 }, { x: 600, y: 400, class_id: 1 }, { x: 750, y: 350, class_id: 1 }
  ]
  const [knnPoints, setKnnPoints] = useState(baseKnnData)

  // Tab 3: Polynomial State
  const [polyPoints, setPolyPoints] = useState([])
  const [polyDegree, setPolyDegree] = useState(3)
  const [polyCurve, setPolyCurve] = useState([])

  // Helper to append professional terminal logs
  const addLog = (type, msg) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [{ time: timestamp, type, msg }, ...prev].slice(0, 5))
  }

  // Dynamic Math Fallback Engine
  const computeLocalLinear = (points) => {
    if (points.length < 2) return { slope: 0, intercept: 0 };
    const n = points.length;
    const sumX = points.reduce((sum, p) => sum + p.x, 0);
    const sumY = points.reduce((sum, p) => sum + p.y, 0);
    const sumXY = points.reduce((sum, p) => sum + p.x * p.y, 0);
    const sumX2 = points.reduce((sum, p) => sum + p.x * p.x, 0);
    
    const num = (n * sumXY) - (sumX * sumY);
    const den = (n * sumX2) - (sumX * sumX);
    if (den === 0) return { slope: 0, intercept: 0 };
    
    const m = num / den;
    const b = (sumY - m * sumX) / n;
    return { slope: m, intercept: b };
  }

  const currentLocalMath = computeLocalLinear(linearPoints);
  const effectiveSlope = regressionData.slope !== 0 ? regressionData.slope : currentLocalMath.slope;
  const effectiveIntercept = regressionData.intercept !== 0 ? regressionData.intercept : currentLocalMath.intercept;

  const displaySlope = isEdgeMode ? parseFloat(effectiveSlope.toFixed(1)) : effectiveSlope;
  const displayIntercept = isEdgeMode ? parseFloat(effectiveIntercept.toFixed(0)) : effectiveIntercept;

  // Live Coefficient Engine ($R^2$ Fitness Score Calculator)
  const calculateR2 = () => {
    if (linearPoints.length < 3) return "0.000";
    const yValues = linearPoints.map(p => p.y);
    const yMean = yValues.reduce((a, b) => a + b, 0) / yValues.length;
    const ssTot = yValues.reduce((acc, y) => acc + Math.pow(y - yMean, 2), 0);
    const ssRes = linearPoints.reduce((acc, p) => {
      const predY = (displaySlope * p.x) + displayIntercept;
      return acc + Math.pow(p.y - predY, 2);
    }, 0);
    if (ssTot === 0) return "1.000";
    const r2 = 1 - (ssRes / ssTot);
    return Math.max(0, Math.min(1, r2)).toFixed(3);
  };

  // --- API CALLS ---
  const fetchLinear = async (points) => {
    if (points.length < 2) return;
    try {
      const res = await axios.post('http://127.0.0.1:8000/api/linear-regression', { points })
      if(res.data && res.data.slope !== undefined) {
        setRegressionData(res.data)
        addLog('API', `Synchronized linear weights with centralized Python engine matrix.`)
      }
    } catch(err) { 
      addLog('WARN', 'External API unreachable. Engaging Edge local analytics core.')
    }
  }

  const fetchPoly = async (points, degree) => {
    if (points.length < 2) return;
    try {
      const res = await axios.post('http://127.0.0.1:8000/api/polynomial-regression', { points, degree })
      setPolyCurve(res.data.curve)
      addLog('API', `Computed high-order polynomial convergence tensor curve at degree ${degree}.`)
    } catch(err) { console.error(err); }
  }

  const fetchKnnBoundary = async (points, k) => {
    try {
      const res = await axios.post('http://127.0.0.1:8000/api/knn-boundary', { points, k })
      setKnnBoundary(res.data.grid)
      addLog('API', `Recalculated discrete proximity grid boundaries using K=${k} nearest neighbors.`)
    } catch(err) { console.error(err); }
  }

  // --- CLICK HANDLERS ---
  const handleCanvasClick = (e) => {
    const rect = svgRef.current.getBoundingClientRect()
    const rawX = e.clientX - rect.left
    const rawY = e.clientY - rect.top
    
    const noiseAmount = simulateNoise ? 70 : 0; 
    const finalX = rawX + (Math.random() - 0.5) * noiseAmount;
    const finalY = rawY + (Math.random() - 0.5) * noiseAmount;
    
    if (activeTab === 'linear') {
      const newPts = [...linearPoints, { x: finalX, y: finalY }]
      setLinearPoints(newPts)
      fetchLinear(newPts)
      addLog('DATA', `Injected coordinate from input vector feed: X:${finalX.toFixed(0)}, Y:${finalY.toFixed(0)}`)
    } 
    else if (activeTab === 'poly') {
      const newPts = [...polyPoints, { x: finalX, y: finalY }]
      setPolyPoints(newPts)
      fetchPoly(newPts, polyDegree)
    }
    else if (activeTab === 'knn') {
      const newPts = [...knnPoints, { x: finalX, y: finalY, class_id: finalX < 400 ? 0 : 1 }]
      setKnnPoints(newPts)
      fetchKnnBoundary(newPts, kValue)
    }
  }

  // Handle Manual Data Entry Submission with Model Bounds Safety Guardrail
  const handleAddManualPoint = () => {
    const x = parseFloat(manualX);
    const y = parseFloat(manualY);
    if(isNaN(x) || isNaN(y)) return alert("Please enter valid numbers!");
    
    // 🚀 NEW: Anomaly Guardrail Logic
    if (x > 1200 || y > 600 || x < -200 || y < -200) {
      addLog('CRIT', `Guardrail Breached: Suppressed extreme matrix outlier to prevent coefficient distortion.`);
      return alert("Outlier Detection Triggered! Value exceeds safe physical device thresholds.");
    }

    if (activeTab === 'linear') {
      const newPts = [...linearPoints, { x, y }]
      setLinearPoints(newPts)
      fetchLinear(newPts)
    } else if (activeTab === 'poly') {
      const newPts = [...polyPoints, { x, y }]
      setPolyPoints(newPts)
      fetchPoly(newPts, polyDegree)
    } else if (activeTab === 'knn') {
      const newPts = [...knnPoints, { x, y, class_id: x < 400 ? 0 : 1 }]
      setKnnPoints(newPts)
      fetchKnnBoundary(newPts, kValue)
    }
    addLog('DATA', `Manual vector parameter injected successfully: X:${x}, Y:${y}`)
    setShowDataModal(false);
    setManualX('');
    setManualY('');
  }

  // CSV UPLOAD PARSER
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const rows = text.trim().split('\n').slice(1);

      const parsedPoints = rows.map(row => {
        const cols = row.split(',');
        return {
          x: parseFloat(cols[0]),
          y: parseFloat(cols[1]),
          class_id: cols[2] ? parseInt(cols[2]) : (parseFloat(cols[0]) < 400 ? 0 : 1)
        };
      }).filter(p => !isNaN(p.x) && !isNaN(p.y));

      if (parsedPoints.length === 0) return alert("Invalid or empty CSV file!");

      if (activeTab === 'linear') {
        setLinearPoints(parsedPoints);
        fetchLinear(parsedPoints);
      } else if (activeTab === 'poly') {
        setPolyPoints(parsedPoints);
        fetchPoly(parsedPoints, polyDegree);
      } else if (activeTab === 'knn') {
        setKnnPoints(parsedPoints);
        fetchKnnBoundary(parsedPoints, kValue);
      }
      addLog('SYSTEM', `Parsed and ingestion completed for ${parsedPoints.length} structural records from CSV stream.`)
    };
    reader.readAsText(file);
    e.target.value = null; 
  }

  // BATCH PROCESS SIMULATOR
  const runBatchProcess = () => {
    const batchPoints = [];
    for(let i=0; i < 35; i++) {
      batchPoints.push({
        x: Math.random() * 650 + 80,
        y: Math.random() * 320 + 80,
        class_id: Math.random() > 0.5 ? 1 : 0
      });
    }
    
    if (activeTab === 'linear') {
      const newPts = [...linearPoints, ...batchPoints]
      setLinearPoints(newPts)
      fetchLinear(newPts)
    } else if (activeTab === 'poly') {
      const newPts = [...polyPoints, ...batchPoints]
      setPolyPoints(newPts)
      fetchPoly(newPts, polyDegree)
    } else if (activeTab === 'knn') {
      const newPts = [...knnPoints, ...batchPoints]
      setKnnPoints(newPts)
      fetchKnnBoundary(newPts, kValue)
    }
    addLog('SYSTEM', `Executed localized batch matrix array training sequence. Appended 35 asynchronous coordinates.`)
  }

  const injectPoison = () => {
    const poisonedPoints = [...linearPoints, { x: 100, y: 420 }]
    setLinearPoints(poisonedPoints)
    fetchLinear(poisonedPoints)
    addLog('CRIT', `Data Corruption Injected! Outlier introduced at coordinate vector [100, 420].`)
  }

  const clearBoard = () => {
    if (activeTab === 'linear') { setLinearPoints([]); setRegressionData({ slope: 0, intercept: 0 }) }
    if (activeTab === 'poly') { setPolyPoints([]); setPolyCurve([]) }
    if (activeTab === 'knn') { setKnnPoints(baseKnnData); fetchKnnBoundary(baseKnnData, kValue) }
    addLog('SYSTEM', `Flushed memory registers. Workspace cleared for next telemetry frame.`)
  }

  const downloadCSV = () => {
    let pointsToExport = activeTab === 'linear' ? linearPoints : (activeTab === 'poly' ? polyPoints : knnPoints);
    if (pointsToExport.length === 0) return alert("No data to export!");
    let csvContent = "data:text/csv;charset=utf-8,X_Coordinate,Y_Coordinate,Class_ID\n" + 
      pointsToExport.map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)},${p.class_id ?? 'N/A'}`).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "edge_ml_dataset.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    addLog('SYSTEM', `Export compiled successfully. Downloaded local dataset filesystem token.`)
  }

  useEffect(() => {
    if (activeTab === 'knn') fetchKnnBoundary(knnPoints, kValue);
  }, [kValue, activeTab])

  useEffect(() => {
    if (activeTab === 'poly') fetchPoly(polyPoints, polyDegree);
  }, [polyDegree])

  // Monitor toggle modes in our rolling terminal
  useEffect(() => {
    addLog('HARDWARE', `Bit-shift Precision Configuration optimized to: ${isEdgeMode ? '8-bit Floating Quantization' : '32-bit Floating Native Precision'}`);
  }, [isEdgeMode])

  const lineStartY = (displaySlope * 0) + displayIntercept
  const lineEndY = (displaySlope * 1000) + displayIntercept

  const polyPathData = polyCurve.length > 0 ? polyCurve.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') : "";

  const generateCppCode = () => {
    return `// Auto-Generated Edge ML Model\n// Ready for deployment to ESP32 / Microcontroller\n\nconst float MODEL_SLOPE = ${displaySlope.toFixed(4)};\nconst float MODEL_INTERCEPT = ${displayIntercept.toFixed(4)};\n\nfloat predict(float sensor_input) {\n    return (MODEL_SLOPE * sensor_input) + MODEL_INTERCEPT;\n}\n\nvoid setup() {\n    Serial.begin(115200);\n    Serial.println("Edge ML Model Booted.");\n}\n\nvoid loop() {\n    delay(1000);\n}`;
  }

  const getActiveData = () => {
    if (activeTab === 'linear') return linearPoints;
    if (activeTab === 'poly') return polyPoints;
    return knnPoints;
  }

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: "'Inter', sans-serif", backgroundColor: '#0f172a', position: 'relative', color: '#f8fafc' }}>
      
      {/* --- SLEEK DARK SIDEBAR (Vertical Padding Shrunk to 15px to completely remove Scrollbars) --- */}
      <div style={{ width: '280px', backgroundColor: '#1e293b', padding: '15px 25px', display: 'flex', flexDirection: 'column', zIndex: 10, borderRight: '1px solid #334155', boxShadow: '4px 0 15px rgba(0,0,0,0.3)' }}>
        <h2 style={{ fontSize: '22px', fontWeight: '800', borderBottom: '1px solid #334155', paddingBottom: '12px', marginTop: 5, color: '#38bdf8', letterSpacing: '1px' }}>
          ⚡ EDGE AI LAB
        </h2>
        
        {/* System Health Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px', marginBottom: '15px' }}>
          <div style={{ width: '10px', height: '10px', backgroundColor: '#10b981', borderRadius: '50%', marginRight: '10px', boxShadow: '0 0 8px #10b981' }}></div>
          <span style={{ fontSize: '12px', fontWeight: 'bold' }}>Hardware System Online</span>
        </div>

        <p style={{ color: '#64748b', fontSize: '11px', marginTop: '10px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1.5px', fontWeight: 'bold' }}>Select Algorithm</p>
        
        <button onClick={() => setActiveTab('linear')} style={{ padding: '12px', textAlign: 'left', backgroundColor: activeTab === 'linear' ? '#0ea5e9' : '#0f172a', color: 'white', border: activeTab === 'linear' ? 'none' : '1px solid #334155', borderRadius: '8px', cursor: 'pointer', marginBottom: '10px', fontSize: '14px', fontWeight: '600', transition: 'all 0.2s' }}>📈 Linear Regression</button>
        <button onClick={() => setActiveTab('poly')} style={{ padding: '12px', textAlign: 'left', backgroundColor: activeTab === 'poly' ? '#0ea5e9' : '#0f172a', color: 'white', border: activeTab === 'poly' ? 'none' : '1px solid #334155', borderRadius: '8px', cursor: 'pointer', marginBottom: '10px', fontSize: '14px', fontWeight: '600', transition: 'all 0.2s' }}>🎢 Overfitting Sandbox</button>
        <button onClick={() => setActiveTab('knn')} style={{ padding: '12px', textAlign: 'left', backgroundColor: activeTab === 'knn' ? '#0ea5e9' : '#0f172a', color: 'white', border: activeTab === 'knn' ? 'none' : '1px solid #334155', borderRadius: '8px', cursor: 'pointer', fontSize: '14px', fontWeight: '600', transition: 'all 0.2s' }}>🎯 KNN Map</button>

        {/* Deployment Target */}
        <div style={{ marginTop: '12px' }}>
          <p style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 'bold' }}>Deployment Target</p>
          <select style={{ width: '100%', padding: '10px', backgroundColor: '#0f172a', color: '#38bdf8', border: '1px solid #334155', borderRadius: '6px', fontWeight: 'bold', outline: 'none' }}>
            <option>ESP32-CAM (IoT)</option>
            <option>Arduino Nano</option>
            <option>STM32 (Industrial)</option>
          </select>
        </div>

        {/* HARDWARE PROFILER */}
        <div style={{ marginTop: '15px', padding: '15px', backgroundColor: '#0f172a', borderRadius: '10px', border: '1px solid #334155', boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.5)' }}>
          <p style={{ color: '#10b981', fontSize: '11px', margin: '0 0 12px 0', textTransform: 'uppercase', fontWeight: '800', letterSpacing: '1px' }}>💻 Live Profiler</p>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: '#94a3b8', fontSize: '12px' }}>Est. RAM:</span>
            <span style={{ color: '#38bdf8', fontSize: '12px', fontFamily: 'monospace', fontWeight: 'bold' }}>{activeTab === 'linear' ? (isEdgeMode ? '2 Bytes' : '8 Bytes') : activeTab === 'knn' ? `${knnPoints.length * (isEdgeMode ? 3 : 12)} Bytes` : `${polyPoints.length * polyDegree * (isEdgeMode ? 1 : 4)} Bytes`}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ color: '#94a3b8', fontSize: '12px' }}>Inference:</span>
            <span style={{ color: '#f43f5e', fontSize: '12px', fontFamily: 'monospace', fontWeight: 'bold' }}>{activeTab === 'linear' ? (isEdgeMode ? '4 µs' : '12 µs') : activeTab === 'knn' ? `${knnPoints.length * (isEdgeMode ? 2 : 5)} µs` : `${polyDegree * 10} µs`}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#94a3b8', fontSize: '12px' }}>Power Draw:</span>
            <span style={{ color: isEdgeMode ? '#10b981' : '#fbbf24', fontSize: '12px', fontFamily: 'monospace', fontWeight: 'bold' }}>{isEdgeMode ? 'Low (15mW)' : 'High (65mW)'}</span>
          </div>
        </div>

        <div style={{ marginTop: 'auto', backgroundColor: '#0f172a', padding: '15px', borderRadius: '10px', border: '1px solid #334155' }}>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '13px', marginBottom: '12px', fontWeight: '600' }}>
            <input type="checkbox" checked={isEdgeMode} onChange={(e) => setIsEdgeMode(e.target.checked)} style={{ marginRight: '12px', width: '16px', height: '16px', accentColor: '#10b981' }}/>
            Edge Sim (8-bit)
          </label>
          <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '13px', color: '#cbd5e1', fontWeight: '600' }}>
            <input type="checkbox" checked={simulateNoise} onChange={(e) => setSimulateNoise(e.target.checked)} style={{ marginRight: '12px', width: '16px', height: '16px', accentColor: '#38bdf8' }}/>
            Sensor Noise
          </label>
        </div>
      </div>

      {/* --- MAIN CONTENT AREA --- */}
      <div style={{ flex: 1, padding: '40px', overflowY: 'auto' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '30px' }}>
          <div>
            <h1 style={{ margin: 0, color: '#f8fafc', fontSize: '32px', fontWeight: '800', letterSpacing: '-0.5px' }}>
              {activeTab === 'linear' ? 'Linear Regression' : activeTab === 'poly' ? 'Polynomial Regression' : 'KNN Territory Map'}
            </h1>
            <p style={{ margin: '8px 0 0 0', color: '#94a3b8', fontSize: '15px' }}>
              {activeTab === 'linear' && 'Train the AI to find the optimal trendline through the data.'}
              {activeTab === 'poly' && 'Increase complexity to watch the algorithm overfit to the noise.'}
              {activeTab === 'knn' && 'Visualize decision boundaries as the AI categorizes the territories.'}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'flex-end', maxWidth: '650px' }}>
            <input type="file" id="csv-upload" accept=".csv" style={{ display: 'none' }} onChange={handleFileUpload} />
            <button onClick={() => document.getElementById('csv-upload').click()} style={{ padding: '10px 18px', backgroundColor: '#64748b', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '800', fontSize: '13px', boxShadow: '0 4px 14px rgba(100, 116, 139, 0.4)' }}>📂 Upload CSV</button>
            <button onClick={() => setShowDataModal(true)} style={{ padding: '10px 18px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '800', fontSize: '13px', boxShadow: '0 4px 14px rgba(59, 130, 246, 0.4)' }}>➕ Enter Data</button>
            <button onClick={runBatchProcess} style={{ padding: '10px 18px', backgroundColor: '#6d28d9', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '800', fontSize: '13px', boxShadow: '0 4px 14px rgba(109, 40, 217, 0.4)' }}>⚡ Batch Process</button>
            <button onClick={downloadCSV} style={{ padding: '10px 18px', backgroundColor: '#1e293b', color: '#38bdf8', border: '1px solid #38bdf8', borderRadius: '8px', cursor: 'pointer', fontWeight: '700', fontSize: '13px' }}>📊 Export CSV</button>
            
            {activeTab === 'linear' && <button onClick={() => setShowCppModal(true)} style={{ padding: '10px 18px', backgroundColor: '#10b981', color: '#022c22', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '800', fontSize: '13px', boxShadow: '0 4px 14px rgba(16, 185, 129, 0.4)' }}>💻 Export C++</button>}
            
            {activeTab === 'linear' && <button onClick={injectPoison} style={{ padding: '10px 18px', backgroundColor: '#f59e0b', color: '#451a03', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '800', fontSize: '13px' }}>🧪 Poison Data</button>}
            <button onClick={clearBoard} style={{ padding: '10px 18px', backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: '700', fontSize: '13px' }}>Clear Canvas</button>
          </div>
        </div>

        {isEdgeMode && <div style={{ backgroundColor: 'rgba(234, 179, 8, 0.1)', border: '1px solid #eab308', color: '#fde047', padding: '15px 20px', borderRadius: '8px', marginBottom: '25px', fontWeight: '600', display: 'flex', alignItems: 'center' }}><span style={{ fontSize: '20px', marginRight: '10px' }}>⚠️</span> EDGE AI ACTIVE: Precision highly reduced to simulate microcontroller memory constraints.</div>}

        {/* --- DARK NEON GRAPH CANVAS --- */}
        <div style={{ border: '1px solid #334155', borderRadius: '12px', overflow: 'hidden', cursor: 'crosshair', backgroundColor: '#1e293b', boxShadow: '0 10px 30px rgba(0,0,0,0.5)', position: 'relative' }}>
          <svg ref={svgRef} width="100%" height="500" onClick={handleCanvasClick} style={{ display: 'block' }}>
            {[...Array(30)].map((_, i) => <line key={`v-${i}`} x1={i * 50} y1="0" x2={i * 50} y2="600" stroke="#232f45" strokeWidth="1" />)}
            {[...Array(15)].map((_, i) => <line key={`h-${i}`} x="0" y1={i * 50} x2="1500" y2={i * 50} stroke="#232f45" strokeWidth="1" />)}
            
            {activeTab === 'knn' && knnBoundary.map((rect, i) => <rect key={`bg-${i}`} x={rect.x} y={rect.y} width="40" height="40" fill={rect.class_id === 0 ? "rgba(56, 189, 248, 0.1)" : "rgba(244, 63, 94, 0.1)"} />)}
            
            {/* Residual Error Plotter */}
            {activeTab === 'linear' && linearPoints.length >= 2 && linearPoints.map((p, i) => (
               <line key={`err-${i}`} x1={p.x} y1={p.y} x2={p.x} y2={(displaySlope * p.x) + displayIntercept} stroke="#ef4444" strokeDasharray="6,4" strokeWidth="2" style={{ filter: 'drop-shadow(0 0 4px rgba(239,68,68,0.8))' }} />
            ))}

            {/* Regression Line and Data Points */}
            {activeTab === 'linear' && linearPoints.length >= 2 && <line x1="0" y1={lineStartY} x2="1500" y2={lineEndY} stroke={isEdgeMode ? "#f43f5e" : "#10b981"} strokeWidth={isEdgeMode ? "8" : "4"} strokeDasharray={isEdgeMode ? "15,15" : "0"} style={{ filter: isEdgeMode ? 'none' : 'drop-shadow(0 0 8px rgba(16,185,129,0.6))' }}/>}
            {activeTab === 'linear' && linearPoints.map((pt, i) => <circle key={`l-${i}`} cx={pt.x} cy={pt.y} r="7" fill="#38bdf8" style={{ filter: 'drop-shadow(0 0 6px rgba(56,189,248,0.8))' }} />)}
            
            {activeTab === 'poly' && polyPoints.length >= 2 && <path d={polyPathData} fill="none" stroke={isEdgeMode ? "#f43f5e" : "#a855f7"} strokeWidth={isEdgeMode ? "8" : "4"} strokeDasharray={isEdgeMode ? "15,15" : "0"} style={{ filter: isEdgeMode ? 'none' : 'drop-shadow(0 0 8px rgba(168,85,247,0.6))' }}/>}
            {activeTab === 'poly' && polyPoints.map((pt, i) => <circle key={`p-${i}`} cx={pt.x} cy={pt.y} r="7" fill="#f43f5e" style={{ filter: 'drop-shadow(0 0 6px rgba(244,63,94,0.8))' }} />)}
            
            {activeTab === 'knn' && knnPoints.map((pt, i) => <circle key={`k-${i}`} cx={pt.x} cy={pt.y} r="9" fill={pt.class_id === 0 ? "#38bdf8" : "#f43f5e"} stroke="#0f172a" strokeWidth="2" style={{ filter: pt.class_id === 0 ? 'drop-shadow(0 0 6px rgba(56,189,248,0.8))' : 'drop-shadow(0 0 6px rgba(244,63,94,0.8))' }} />)}

            {/* Calibration Grid Markers */}
            <text x="15" y="480" fill="#475569" fontSize="10" fontFamily="monospace" fontWeight="bold">Y: 0.00V (GND)</text>
            <text x="15" y="25" fill="#475569" fontSize="10" fontFamily="monospace" fontWeight="bold">Y: 5.00V (VCC)</text>
            <text x="8" y="440" fill="#475569" fontSize="9" fontFamily="monospace" transform="rotate(-90 8 440)" style={{ letterSpacing: '1.5px', fontWeight: 'bold' }}>▲ TELEMETRY AMPLITUDE</text>
            <text x="140" y="480" fill="#475569" fontSize="10" fontFamily="monospace" fontWeight="bold">X: 100 Hz</text>
            <text x="750" y="480" fill="#475569" fontSize="10" fontFamily="monospace" fontWeight="bold">X: 2.4 GHz</text>
          </svg>
        </div>

        {/* --- TELEMETRY READOUTS --- */}
        {activeTab === 'linear' && (
           <div style={{ marginTop: '25px', padding: '25px', backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', display: 'flex', justifyContent: 'space-between', gap: '20px', boxShadow: '0 4px 20px rgba(0,0,0,0.2)' }}>
             <div style={{ flex: 1 }}><p style={{ color: '#94a3b8', fontSize: '11px', margin: '0 0 8px 0', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>Slope (m)</p><h2 style={{ margin: 0, color: isEdgeMode ? '#f43f5e' : '#38bdf8', fontFamily: 'monospace', fontSize: '24px' }}>{displaySlope.toFixed(3)}</h2></div>
             <div style={{ flex: 1 }}><p style={{ color: '#94a3b8', fontSize: '11px', margin: '0 0 8px 0', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>Intercept (b)</p><h2 style={{ margin: 0, color: isEdgeMode ? '#f43f5e' : '#38bdf8', fontFamily: 'monospace', fontSize: '24px' }}>{displayIntercept.toFixed(3)}</h2></div>
             <div style={{ flex: 1, borderLeft: '1px solid #334155', paddingLeft: '25px' }}><p style={{ color: '#10b981', fontSize: '11px', margin: '0 0 8px 0', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>Model Fitness (R² Score)</p><h2 style={{ margin: 0, color: '#10b981', fontFamily: 'monospace', fontSize: '24px' }}>{calculateR2()}</h2></div>
           </div>
        )}
        
        {activeTab === 'poly' && (
           <div style={{ marginTop: '25px', padding: '25px', backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', boxShadow: '0 4px 20px rgba(0,0,0,0.2)' }}>
             <p style={{ color: '#e2e8f0', fontSize: '15px', margin: '0 0 15px 0', fontWeight: '600' }}>AI Complexity Layer: <span style={{ color: '#a855f7', fontSize: '20px' }}>{polyDegree}</span></p>
             <input type="range" min="1" max="12" step="1" value={polyDegree} onChange={(e) => setPolyDegree(parseInt(e.target.value))} style={{ width: '400px', accentColor: '#a855f7' }}/>
           </div>
        )}

        {activeTab === 'knn' && (
          <div style={{ marginTop: '25px', padding: '25px', backgroundColor: '#1e293b', borderRadius: '12px', border: '1px solid #334155', boxShadow: '0 4px 20px rgba(0,0,0,0.2)' }}>
            <p style={{ color: '#e2e8f0', fontSize: '15px', margin: '0 0 15px 0', fontWeight: '600' }}>Decision Neighbors (K-Value): <span style={{ color: '#38bdf8', fontSize: '20px' }}>{kValue}</span></p>
            <input type="range" min="1" max="15" step="2" value={kValue} onChange={(e) => setKValue(parseInt(e.target.value))} style={{ width: '400px', accentColor: '#38bdf8' }}/>
          </div>
        )}

        {/* UNIVERSAL INFERENCE MONITOR TABLE */}
        <div style={{ marginTop: '20px', backgroundColor: '#020617', padding: '20px', borderRadius: '12px', border: '1px solid #334155', boxShadow: 'inset 0 2px 10px rgba(0,0,0,0.5)' }}>
          <p style={{ color: '#10b981', fontSize: '11px', fontWeight: '800', margin: '0 0 15px 0', textTransform: 'uppercase', letterSpacing: '1px' }}>📡 Live Inference Feed: {activeTab.toUpperCase()} Engine</p>
          <table style={{ width: '100%', color: '#e2e8f0', fontSize: '13px', textAlign: 'center', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                <th style={{ paddingBottom: '10px' }}>Input (X)</th>
                <th style={{ paddingBottom: '10px' }}>Actual (Y)</th>
                <th style={{ paddingBottom: '10px' }}>{activeTab === 'knn' ? 'Predicted Class' : 'Predicted Output'}</th>
              </tr>
            </thead>
            <tbody>
              {getActiveData().slice(-4).map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ paddingTop: '10px', paddingBottom: '10px', fontFamily: 'monospace' }}>{p.x.toFixed(1)}</td>
                  <td style={{ paddingTop: '10px', paddingBottom: '10px', fontFamily: 'monospace' }}>{p.y.toFixed(1)}</td>
                  <td style={{ paddingTop: '10px', paddingBottom: '10px', color: '#10b981', fontFamily: 'monospace', fontWeight: 'bold' }}>
                    {activeTab === 'linear' ? ((displaySlope * p.x) + displayIntercept).toFixed(1) : 
                     activeTab === 'knn' ? (p.class_id === 0 ? 'Class 0 (Blue)' : 'Class 1 (Red)') : 
                     'Fitted to Curve'}
                  </td>
                </tr>
              ))}
              {getActiveData().length === 0 && (
                <tr><td colSpan="3" style={{ padding: '20px', color: '#64748b', fontStyle: 'italic' }}>Awaiting sensor data... Click the canvas or run a batch cycle to begin.</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* 🚀 NEW: ROLLING SYSTEM ACTIVITY TELEMETRY LOGS (Console Station) */}
        <div style={{ marginTop: '20px', backgroundColor: '#090d16', padding: '15px 20px', borderRadius: '12px', border: '1px solid #1e293b', boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.8)' }}>
          <p style={{ color: '#a855f7', fontSize: '11px', fontWeight: '800', margin: '0 0 12px 0', textTransform: 'uppercase', letterSpacing: '1px' }}>📟 Real-Time System Terminal Logs</p>
          <div style={{ fontFamily: 'monospace', fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '110px', overflowY: 'hidden' }}>
            {logs.map((log, index) => (
              <div key={index} style={{ color: log.type === 'CRIT' ? '#ef4444' : log.type === 'WARN' ? '#f59e0b' : log.type === 'DATA' ? '#38bdf8' : '#94a3b8', opacity: 1 - index * 0.15 }}>
                <span style={{ color: '#475569', marginRight: '8px' }}>[{log.time}]</span>
                <span style={{ fontWeight: 'bold', marginRight: '6px' }}>[{log.type}]</span>
                {log.msg}
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* MANUAL TELEMETRY DATA ENTRY MODAL (With Integrated Outlier Guardrail Alerts) */}
      {showDataModal && (
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.85)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 100, backdropFilter: 'blur(4px)' }}>
          <div style={{ backgroundColor: '#0f172a', padding: '35px', borderRadius: '16px', width: '400px', border: '1px solid #3b82f6', boxShadow: '0 0 30px rgba(59,130,246,0.2)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ color: '#38bdf8', margin: 0, fontWeight: '800' }}>Inject Data Point</h2>
              <button onClick={() => setShowDataModal(false)} style={{ backgroundColor: 'transparent', border: 'none', color: '#64748b', fontSize: '24px', cursor: 'pointer' }}>✖</button>
            </div>
            <p style={{ color: '#94a3b8', fontSize: '13px', marginBottom: '20px' }}>Manually enter exact coordinates. Systems safety bounds are currently set to X: [-200 to 1200], Y: [-200 to 600].</p>
            
            <div style={{ display: 'flex', gap: '15px', marginBottom: '25px' }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', color: '#64748b', fontSize: '11px', marginBottom: '8px', fontWeight: 'bold' }}>X COORDINATE</label>
                <input type="number" placeholder="e.g. 250" value={manualX} onChange={e => setManualX(e.target.value)} style={{ width: '100%', padding: '12px', backgroundColor: '#1e293b', border: '1px solid #334155', color: 'white', borderRadius: '8px', boxSizing: 'border-box', outline: 'none' }} />
              </div>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', color: '#64748b', fontSize: '11px', marginBottom: '8px', fontWeight: 'bold' }}>Y COORDINATE</label>
                <input type="number" placeholder="e.g. 150" value={manualY} onChange={e => setManualY(e.target.value)} style={{ width: '100%', padding: '12px', backgroundColor: '#1e293b', border: '1px solid #334155', color: 'white', borderRadius: '8px', boxSizing: 'border-box', outline: 'none' }} />
              </div>
            </div>
            
            <button onClick={handleAddManualPoint} style={{ width: '100%', padding: '14px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', fontWeight: '800', cursor: 'pointer', fontSize: '14px' }}>Submit Data Point</button>
          </div>
        </div>
      )}

      {/* C++ INTERFACE MODAL EXPORT */}
      {showCppModal && (
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.85)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 100, backdropFilter: 'blur(4px)' }}>
          <div style={{ backgroundColor: '#0f172a', padding: '35px', borderRadius: '16px', width: '650px', border: '1px solid #10b981', boxShadow: '0 0 30px rgba(16,185,129,0.2)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ color: '#10b981', margin: 0, fontWeight: '800' }}>C++ Hardware Export</h2>
              <button onClick={() => setShowCppModal(false)} style={{ backgroundColor: 'transparent', border: 'none', color: '#64748b', fontSize: '24px', cursor: 'pointer' }}>✖</button>
            </div>
            <p style={{ color: '#94a3b8', fontSize: '14px', marginBottom: '20px' }}>This C++ code has been generated live based on your AI math. It is ready to be flashed to an ESP32 or Arduino.</p>
            <textarea readOnly value={generateCppCode()} style={{ width: '100%', height: '300px', backgroundColor: '#020617', color: '#10b981', fontFamily: 'monospace', padding: '20px', borderRadius: '8px', border: '1px solid #334155', boxSizing: 'border-box', outline: 'none', fontSize: '14px' }} />
          </div>
        </div>
      )}
    </div>
  )
}

export default App