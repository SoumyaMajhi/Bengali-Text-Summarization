
# 🧠 Bengali Text Summarizer using Transformer + FastAPI

A powerful Bengali text summarization web app built with a custom Transformer model and served using FastAPI. Input Bengali text and receive a concise, generated summary powered by a deep learning model.

---

## 🚀 Features

- ✍️ **Summarize Bengali Text**: Generate concise summaries for long-form Bengali input.
- ⚙️ **Transformer-based Model**: Custom-built encoder-decoder architecture with multi-head attention.
- 🧠 **Custom-trained SentencePiece Tokenizer**: Efficient subword tokenization for Bengali.
- 🌐 **FastAPI Backend**: Fast, asynchronous API with a responsive web interface.
- 💾 **Model Checkpoint Loader**: Automatically loads the latest trained weights.
- ✅ **Health Check Route**: Useful for deployment uptime checks.

---

## 📸 Demo

> Coming soon! 

---

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow 2.x** - Deep learning framework
- **SentencePiece** - Tokenizer for Bengali text
- **FastAPI** - Web framework for serving the app
- **Jinja2** - HTML template rendering
- **HTML/CSS/JavaScript/Bootstrap** - Frontend for user input and display
- **[Render](https://render.com/)** - For Deployment

---

## 📂 Project Structure

```
.
├── main.py                  # FastAPI app
├── transformer.py           # Model architecture and summarization logic
├── tokenizer/               # SentencePiece model
│   └── bengali_spm.model
├── weights/                 # Trained model checkpoints
├── templates/
│   └── index.html           # Frontend page
├── static/                  # CSS, JS files (optional)
└── README.md                
```

---

## ⚡ Quick Start

### 📁 1. Clone the repository
```b
git clone https://github.com/SoumyaMajhi/Bengali-Text-Summarization.git
cd Bengali-Text-Summarization
```

### 🧪 2. Create and activate a virtual environment (Optional)
```
python -m venv venv
```
### ▶️ Activate the environment:
On Windows:
```
venv\Scripts\activate
```
On macOS/Linux:
```
source venv/bin/activate
```

### 🔧 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚦 4. Run Locally

```bash
uvicorn main:app --reload
```

🔗 Then open your browser at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔍 API Reference

### `GET /`
Returns the web interface.

### `POST /summarize`

**Input JSON:**
```json
{
  "original_text": "আপনার ইনপুট বাংলা টেক্সট এখানে"
}
```

**Response:**
```json
{
  "summary": "সারাংশ"
}
```

### `GET /health`
Check if the server is running.

---

## 📚 Model Details

- Transformer with:
  - `NUM_LAYERS = 4`
  - `D_MODEL = 128`
  - `NUM_HEADS = 8`
  - `DFF = 512`
  - `TEXT_LENGTH = 512`
  - `SUMMARY_LENGTH = 16`
  - `VOCAB_SIZE = 10000`

- Trained on two custom Bengali datasets scraped from [ProthomAlo](https://www.kaggle.com/datasets/samym4/prothom-alo-cleaned-dataset) & [Eisamay](https://www.kaggle.com/datasets/samym4/eisamay-bengali-news-dataset)
- Custom Tokenization on our datasets with SentencePiece (`.model` included in `tokenizer/`)

---

## 🔧 For Render Deployment
Set:
- `Build Command` as `pip install -r requirements.txt`
- `Start Command` as `uvicorn main:app --host 0.0.0.0 --port 8000`

Set the following environment variables:
- `PYTHON_VERSION = 3.10.13` (for Tensorflow compatibility)
- `HOST = 0.0.0.0`
- `PORT = 8000`





## 🧠 Future Work

- Add a frontend summarization history
- Upload and summarize `.txt` or `.pdf` files
- Host on Hugging Face
- Model improvement (more layers or pretrained embeddings)
- Dockerized backend

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change or improve.
