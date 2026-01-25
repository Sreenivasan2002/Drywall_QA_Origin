# Prompted Segmentation for Drywall QA

A text-conditioned segmentation model for drywall quality assurance that processes images and natural language prompts (e.g., "segment crack", "segment taping area") to produce binary masks.

## 🎯 Project Overview

This project uses **CLIPSeg** to perform text-prompted segmentation on:
- **Drywall Joint Detection** - Identifies taping areas and wall seams
- **Crack Detection** - Detects cracks and damage in walls

## 📊 Results

| Dataset | IoU Score | Dice Score |
|---------|-----------|------------|
| Drywall | 0.1234 | 0.1888 |
| Cracks | 0.4277 | 0.5744 |
| Overall | 0.1971 | 0.2823 |

**Pixel Accuracy: 90.21%**

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/Sreenivasan2002/Drywall_QA_Origin.git
cd Drywall_QA_Origin
```

2. Install dependencies:
```bash
pip install roboflow transformers torch torchvision opencv-python matplotlib python-dotenv
```

3. Create `.env` file with your Roboflow API key:
```bash
cp .env.example .env
# Edit .env and add your API key
```

4. Download model weights from Google Drive (link below)



## 📁 Project Structure
```
├── origin_assessment.ipynb   # Main Colab notebook
├── final_report.txt          # Evaluation report
├── evaluation_metrics.json   # IoU, Dice scores
├── training_curves.png       # Loss plots
├── predictions_*.png         # Visual results
├── demo_*.png                # Prompted segmentation demos
├── predictions/              # Output masks
├── .env.example              # Environment template
└── README.md
```

## 🔍 Usage
```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Load model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.load_state_dict(torch.load('best_model.pth'))

# Segment with text prompt
inputs = processor(text="crack", images=image, return_tensors="pt")
outputs = model(**inputs)
mask = torch.sigmoid(outputs.logits)
```

