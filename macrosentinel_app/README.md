# MacroSentinel v3 — Flask App

## Local setup (run on your own machine)

1. Copy your trained model file into this folder:
   macrosentinel_model.keras

2. Install dependencies:
   pip install -r requirements.txt

3. Run the server:
   python app.py

4. Open your browser:
   http://127.0.0.1:5000

## Deploy to Render (free public link)

1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service → connect your repo
3. Set:
   - Build command:  pip install -r requirements.txt
   - Start command:  gunicorn app:app
4. Upload macrosentinel_model.keras as an environment file or
   retrain on Render by including macrosentinel_v3_final.py
5. Render gives you a public URL like:
   https://macrosentinel.onrender.com

## Folder structure
macrosentinel_app/
├── app.py                      ← Flask backend (API + routes)
├── templates/
│   └── index.html              ← Dashboard frontend
├── requirements.txt            ← Python dependencies
├── Procfile                    ← For Render/Heroku deployment
├── runtime.txt                 ← Python version
├── macrosentinel_model.keras   ← YOUR TRAINED MODEL (copy here)
└── README.md
