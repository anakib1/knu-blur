# Background Blur Project

This project implements a background blur system with the following components:
- Detector interface for person/background segmentation
- Real-time application for background blur
- Integration with pre-trained models
- Training script for custom model training

## Project Structure
```
background-filter/
├── detector/
│   ├── __init__.py
│   ├── base.py          # Base detector interface
│   └── models/          # Different detector implementations
├── app/
│   ├── __init__.py
│   └── realtime.py      # Real-time application
├── training/
│   ├── __init__.py
│   └── train.py         # Training script
├── utils/
│   ├── __init__.py
│   └── visualization.py # Visualization utilities
├── requirements.txt
└── README.md
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the real-time application:
```bash
python -m app.realtime
```

2. Train your own model:
```bash
python -m training.train
``` 