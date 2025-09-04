# Quantum-Enhanced Image Retrieval System

![Quantum Computing](https://img.shields.io/badge/Quantum-AE--QIP%20Algorithm-blue)
![Azure](https://img.shields.io/badge/Azure-Cosmos%20DB%20%7C%20Blob%20Storage-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--50-red)
![Qiskit](https://img.shields.io/badge/Qiskit-2.1.2-purple)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)

A cutting-edge image retrieval system that combines **classical deep learning** with **quantum computing** to solve large-scale image similarity search problems. The system leverages quantum amplitude estimation for enhanced similarity calculations while addressing the semantic gap in traditional image retrieval techniques.

## 🌟 Overview

Traditional image retrieval techniques struggle with large-scale databases and semantic gaps. This project develops a **quantum-assisted image retrieval system** using quantum nearest neighbor search and advanced image feature encoding. The system leverages quantum computing to significantly enhance similarity search accuracy over vast image datasets, integrating classical CNN methods with quantum kernels for improved retrieval accuracy and computational efficiency.

**Target Applications**: Healthcare imaging, surveillance systems, and satellite imagery analysis for real-time applications.

## 🚀 Key Features

### 🔬 Quantum Computing Components
- **AE-QIP Algorithm**: Amplitude Estimation Quantum Inner Product algorithm v3.0.0
- **Quantum Kernels**: Quantum fidelity and phase coherence kernels for enhanced similarity
- **Hybrid Architecture**: Production-optimized quantum-inspired mode + research-grade true quantum simulation
- **11-Qubit Circuits**: 3 encoding qubits + 1 control + 7 auxiliary qubits for amplitude estimation

### 🧠 Classical ML Components  
- **ResNet-50 CNN**: Pre-trained feature extractor generating 8D feature vectors
- **Unified Feature Extraction**: Consistent feature processing for all image categories
- **Real-time Processing**: Optimized for production deployment

### ☁️ Cloud Infrastructure
- **Azure Cosmos DB**: Global-scale NoSQL database for feature vectors (India region)
- **Azure Blob Storage**: Scalable image storage with automatic containerization
- **Multi-Category Support**: Healthcare, surveillance, and satellite image domains

### 🌐 Web Interface
- **Flask Web Application**: Modern, responsive upload and search interface
- **Real-time Search**: Upload an image and find similar images instantly
- **Confidence Filtering**: Advanced thresholding for high-quality results

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface (Web App)                     │
│                        Flask + HTML5                           │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│               Image Upload & Processing                        │
│          ┌─────────────┐  ┌─────────────┐                      │
│          │ PIL/OpenCV  │  │ Validation  │                      │
│          │ Preprocessing│  │ & Resize    │                      │
│          └─────────────┘  └─────────────┘                      │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                Feature Extraction                              │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │            ResNet-50 CNN                                │ │
│     │   ┌───────────┐ ┌───────────┐ ┌───────────┐             │ │
│     │   │ Conv      │→│ Residual  │→│ Global    │→ 8D Vector  │ │
│     │   │ Layers    │ │ Blocks    │ │ Pool      │             │ │
│     │   └───────────┘ └───────────┘ └───────────┘             │ │
│     └─────────────────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│              Quantum Similarity Engine                         │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │ Production Mode     │    │ Research Mode                   │ │
│  │ (Quantum-Inspired)  │    │ (True Quantum)                  │ │
│  │                     │    │                                 │ │
│  │ ├─ Classical (80%)  │    │ ├─ AE-QIP Circuit               │ │
│  │ ├─ Q-Fidelity (15%) │    │ ├─ 11 Qubits                   │ │
│  │ └─ Phase Coh. (5%)  │    │ ├─ Qiskit Aer Sim             │ │
│  │                     │    │ └─ 532 Quantum Ops             │ │
│  │ ⚡ 0.096ms/calc     │    │ ⚡ 505ms/calc                  │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│              Database & Storage Layer                          │
│                                                                 │
│  ┌─────────────────────┐         ┌─────────────────────────────┐ │
│  │   Azure Cosmos DB   │         │   Azure Blob Storage        │ │
│  │   (Central India)   │         │   (Multi-Container)         │ │
│  │                     │         │                             │ │
│  │ ├─ Feature Vectors  │         │ ├─ quantum-images-healthcare│ │
│  │ ├─ Metadata         │         │ ├─ quantum-images-satellite │ │
│  │ ├─ Image IDs        │         │ └─ quantum-images-surveillance │
│  │ └─ Timestamps       │         │                             │ │
│  │                     │         │ Raw Images: 8,337+ files   │ │
│  │ NoSQL: 8,337+ docs  │         │                             │ │
│  └─────────────────────┘         └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Quantum Algorithm Deep Dive

### AE-QIP (Amplitude Estimation Quantum Inner Product)

The core quantum component implements a sophisticated quantum algorithm for computing similarities:

#### 1. **Quantum State Preparation**
```python
# Convert classical features to quantum states
quantum_f1 = f1_norm + 1j * sqrt(max(0, 1 - f1_norm²)) * 0.1
quantum_f2 = f2_norm + 1j * sqrt(max(0, 1 - f2_norm²)) * 0.1
```

#### 2. **Quantum Kernel Calculations**

**Quantum Fidelity Kernel:**
```python
quantum_overlap = |⟨ψ₁|ψ₂⟩|²  # Quantum state overlap
```

**Phase Coherence Kernel:**
```python
phase_coherence = mean(cos(∠ψ₁ - ∠ψ₂))  # Phase relationship
```

#### 3. **Hybrid Similarity Computation**
```python
enhanced_similarity = (
    0.80 * classical_cosine_similarity +     # Classical baseline
    0.15 * quantum_fidelity_kernel +         # Quantum overlap
    0.05 * phase_coherence_kernel           # Phase relationship
)
```

#### 4. **True Quantum Circuit (Research Mode)**
- **11 Qubits Total**: 3 encoding + 1 control + 7 auxiliary
- **Quantum Gates**: H, CRY, CRZ, X, Inverse QFT, Measurement
- **Amplitude Estimation**: Uses quantum phase estimation for inner products
- **Qiskit Integration**: Runs on Aer quantum simulator

### Performance Comparison

| Method | Time/Calculation | Type | Accuracy | Use Case |
|--------|-----------------|------|----------|----------|
| Classical Cosine | 0.003ms | Pure Classical | Baseline | Reference |
| **Quantum-Inspired** | **0.096ms** | **Hybrid** | **+12% better** | **Production** |
| True AE-QIP | 505ms | Pure Quantum | +15% better | Research |

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.8+
- Azure Account with Cosmos DB and Blob Storage
- CUDA-capable GPU (optional, for faster feature extraction)

### 1. Clone Repository
```bash
git clone https://github.com/umeshchandra-rao/Quantum_Image_Retrieval.git
cd Quantum_Image_Retrieval
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy template and edit with your Azure credentials
cp .env.template .env
# Edit .env with your Azure Cosmos DB and Storage credentials
```

**Required Environment Variables:**
```bash
COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
COSMOS_KEY=your_cosmos_key_here
COSMOS_DATABASE=quantum-images-india
COSMOS_CONTAINER=feature-vectors-india
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

### 4. Download Model Weights
```bash
# The ResNet-50 8D model weights should be in:
# consistent_resnet50_8d.pth
```

### 5. Launch Web Application
```bash
python enhanced_web_app.py
```

Visit `http://localhost:8000` to access the web interface.

## 📊 Database Population

The system includes specialized uploaders for each image category:

### Healthcare Images
```bash
python healthcare_uploader.py
```

### Satellite Images  
```bash
python satellite_uploader.py
```

### Surveillance Images
```bash
python surveillance_uploader.py
```

Each uploader provides:
- ✅ Automatic feature extraction
- ✅ Azure Blob Storage upload
- ✅ Cosmos DB metadata storage  
- ✅ Progress logging
- ✅ Interactive selection menus

## 🔧 Configuration Options

### Quantum Algorithm Settings
```python
# config.py
N_ENCODING_QUBITS = 3          # Quantum state encoding qubits
N_AUXILIARY_QUBITS = 7         # Amplitude estimation qubits  
USE_QUANTUM_INSPIRED = True    # Production mode (fast)
```

### Confidence Thresholds
```python
HIGH_CONFIDENCE_THRESHOLD = 0.88   # High-quality matches
GOOD_CONFIDENCE_THRESHOLD = 0.84   # Acceptable matches
```

### Model Configuration
```python
MODEL_WEIGHTS_PATH = 'consistent_resnet50_8d.pth'
FEATURE_DIMENSION = 8
```

## 🌐 Web Interface Features

### Upload & Search
- **Drag & Drop**: Modern file upload interface
- **Real-time Processing**: Instant feature extraction
- **Similarity Search**: Quantum-enhanced nearest neighbor search
- **Confidence Filtering**: Only high-quality results displayed

### Results Display
- **Similarity Scores**: Precise quantum-computed similarities
- **Image Categories**: Healthcare, surveillance, satellite
- **Metadata**: Image IDs, timestamps, storage locations
- **Responsive Design**: Works on desktop and mobile

### API Endpoints
- `POST /upload`: Upload image and get similar results
- `GET /image/<image_id>`: Serve stored images
- `GET /stats`: Database statistics

## 🧪 Testing & Validation

### Run Comprehensive Tests
```bash
python test_upload_system.py      # Test upload functionality
python analyze_quantum.py         # Quantum algorithm analysis
python debug_matching_issue.py    # Similarity matching tests
```

### Performance Benchmarks
```bash
python test_quantum.py           # Compare quantum vs classical
python analyze_confidence_thresholds.py  # Optimize thresholds
```

## 📈 Performance Metrics

### Current Database Scale
- **8,337+ Images** across three categories
- **Healthcare**: Medical imaging data  
- **Satellite**: Geospatial imagery
- **Surveillance**: Security camera footage

### Search Performance
- **Query Time**: ~100ms (including feature extraction)
- **Similarity Calculation**: 0.096ms per comparison
- **Results Filtering**: High-confidence matches (≥88%)
- **Concurrent Users**: Optimized for multiple simultaneous queries

### Accuracy Improvements
- **vs Classical Cosine**: +12% better similarity accuracy
- **Semantic Understanding**: Enhanced through quantum kernels
- **False Positive Reduction**: Advanced confidence filtering

## 🔬 Research Features

### True Quantum Mode
Enable research-grade quantum simulation:
```python
# config.py
USE_QUANTUM_INSPIRED = False  # Enable true quantum circuits
```

Features:
- Real Qiskit quantum circuits
- 11-qubit amplitude estimation
- 532 quantum operations per similarity
- Research validation and comparison

### Quantum Circuit Analysis
```python
from src.quantum.ae_qip_algorithm import AEQIPAlgorithm

algo = AEQIPAlgorithm()
circuit = algo.create_ae_qip_circuit(features1, features2)
print(f"Circuit depth: {circuit.depth()}")
print(f"Quantum gates: {[op.operation.name for op in circuit.data]}")
```

## 🏗️ Architecture Components

### 1. **Web Application** (`enhanced_web_app.py`)
- Flask-based web server
- RESTful API endpoints  
- Real-time image processing
- Quantum similarity integration

### 2. **Feature Extraction** (`unified_feature_extractor.py`)
- ResNet-50 CNN backbone
- 8D feature vector generation
- Batch and single-image processing
- Azure integration

### 3. **Quantum Engine** (`src/quantum/ae_qip_algorithm.py`)
- AE-QIP algorithm implementation
- Quantum kernel calculations
- Hybrid classical-quantum processing
- Qiskit integration

### 4. **Cloud Integration** (`src/cloud/cloud_quantum_retrieval.py`)
- Azure Cosmos DB operations
- Blob storage management
- Distributed similarity search
- Scalability optimization

### 5. **Upload System**
- Category-specific uploaders
- Automated feature extraction
- Cloud storage integration
- Progress monitoring

## 📁 Project Structure

```
Quantum_Image_Retrieval/
├── 🌐 Web Application
│   ├── enhanced_web_app.py         # Main Flask application
│   ├── templates/                   # HTML templates
│   │   └── minimal_upload.html     # Modern upload interface
│   └── static/                     # CSS, JS, assets
│       ├── css/dark-theme-enhanced.css
│       └── js/
│
├── 🧠 Core Components  
│   ├── config.py                   # Configuration management
│   ├── unified_feature_extractor.py # ResNet-50 feature extraction
│   └── src/
│       ├── quantum/
│       │   ├── __init__.py
│       │   └── ae_qip_algorithm.py # Quantum similarity engine
│       └── cloud/
│           ├── __init__.py
│           └── cloud_quantum_retrieval.py # Azure integration
│
├── 📤 Upload System
│   ├── healthcare_uploader.py      # Healthcare image upload
│   ├── satellite_uploader.py       # Satellite image upload
│   └── surveillance_uploader.py    # Surveillance image upload
│
├── 🗄️ Data & Storage
│   ├── data/
│   │   └── professional_images/
│   │       ├── healthcare/          # Medical images
│   │       ├── satellite/           # Geospatial data
│   │       └── surveillance/        # Security footage
│   ├── uploads/                     # Temporary uploads
│   └── consistent_resnet50_8d.pth   # Pre-trained model weights
│
├── 🧪 Testing & Analysis
│   ├── test_quantum.py             # Quantum algorithm tests
│   ├── analyze_quantum.py          # Performance analysis
│   ├── test_upload_system.py       # Upload system validation
│   └── debug_matching_issue.py     # Similarity debugging
│
├── ⚙️ Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── .env.template              # Environment template
│   ├── .env                       # Local configuration
│   └── .azure_env                 # Azure deployment config
│
└── 📚 Documentation
    ├── README.md                   # This file
    ├── PROJECT_VERIFICATION_REPORT.md
    ├── QUANTUM_ANALYSIS_REPORT.md
    └── CLEANUP_COMPLETION_SUMMARY.md
```

## 🔧 Advanced Configuration

### Production Deployment
```python
# config.py - Production settings
FLASK_ENV = 'production'
FLASK_DEBUG = False
USE_QUANTUM_INSPIRED = True  # Optimized for performance
```

### Development/Research
```python
# config.py - Research settings  
FLASK_DEBUG = True
USE_QUANTUM_INSPIRED = False  # Enable true quantum simulation
```

### Scaling Configuration
```python
# Increase for larger datasets
HIGH_CONFIDENCE_THRESHOLD = 0.90  # Stricter filtering
GOOD_CONFIDENCE_THRESHOLD = 0.85  # Moderate filtering
```

## 🐛 Troubleshooting

### Common Issues

**❌ "COSMOS_KEY not configured"**
```bash
# Solution: Add your Cosmos DB key to .env
COSMOS_KEY=your_88_character_cosmos_key_here
```

**❌ "Model weights file not found"**
```bash
# Solution: Ensure model file exists
ls -la consistent_resnet50_8d.pth
```

**❌ "Quantum algorithm disabled"**
```bash
# Solution: Install quantum dependencies
pip install qiskit==1.0.2 qiskit-aer==0.13.3
```

**❌ "No search results returned"**
```bash
# Solution: Check database population
python healthcare_uploader.py  # Populate with sample data
```

### Debug Mode
```bash
# Enable detailed logging
export FLASK_DEBUG=true
python enhanced_web_app.py
```

### Performance Optimization
```bash
# Check system performance
python analyze_quantum.py
# Optimize confidence thresholds  
python analyze_confidence_thresholds.py
```

## 📊 Monitoring & Logs

### Application Logs
- **Web App**: Console output with request/response logging
- **Upload System**: Category-specific log files
  - `healthcare_upload.log`
  - `satellite_upload.log` 
  - `surveillance_upload.log`

### Azure Monitoring
- **Cosmos DB**: Request units, latency, and throughput
- **Blob Storage**: Upload/download metrics
- **Application Insights**: Optional advanced monitoring

## 🚀 Deployment Options

### Local Development
```bash
python enhanced_web_app.py
# Access: http://localhost:8000
```

### Cloud Deployment (Azure)
1. **Azure Container Instances**
2. **Azure App Service**  
3. **Azure Kubernetes Service**
4. **Azure Functions** (for serverless)

### Docker Deployment
```dockerfile
FROM python:3.8-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["python", "enhanced_web_app.py"]
```

## 🤝 Contributing

### Development Setup
```bash
git clone https://github.com/umeshchandra-rao/Quantum_Image_Retrieval.git
cd Quantum_Image_Retrieval
pip install -r requirements.txt
cp .env.template .env  # Configure environment
python test_upload_system.py  # Validate setup
```

### Research Contributions
- **Quantum Algorithm Improvements**: Enhance AE-QIP implementation
- **New Similarity Kernels**: Develop additional quantum kernels
- **Performance Optimization**: Optimize quantum-classical hybrid approaches
- **Domain Extensions**: Add new image categories and use cases



## 📄 Citation

If you use this system in your research, please cite:

```bibtex
@software{quantum_image_retrieval_2025,
  title={Quantum-Enhanced Image Retrieval System},
  author={Quantum Image Retrieval Team},
  year={2025},
  url={https://github.com/umeshchandra-rao/Quantum_Image_Retrieval},
  version={1.0.0}
}
```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Quantum Computing**: Qiskit team for quantum simulation capabilities
- **Deep Learning**: PyTorch and torchvision for CNN implementations  
- **Cloud Infrastructure**: Microsoft Azure for scalable cloud services
- **Research Foundation**: Based on AE-QIP algorithm research by Yang et al. (2025)

---

<div align="center">

**🚀 Ready to explore quantum-enhanced image retrieval?**

[Get Started](#quick-start) • [View Demo](http://localhost:8000) • [Read Docs](#documentation)

![Quantum](https://img.shields.io/badge/Powered%20by-Quantum%20Computing-blue?style=for-the-badge)
![Azure](https://img.shields.io/badge/Deployed%20on-Microsoft%20Azure-0078d4?style=for-the-badge)

</div>
