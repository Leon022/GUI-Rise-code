### 1. Prerequisites
```bash
conda create -n showui python=3.11
conda activate showui
```

#### 1.1. Clone the Repository
```bash
git clone https://github.com/showlab/ShowUI.git
cd ShowUI
```

#### 1.2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 1.3. Start the Interface
```bash
python app.py
```
If you successfully start the interface, you will see two URLs in the terminal:
```bash
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live (Do not share this link with others, or they will be able to control your computer.)
```