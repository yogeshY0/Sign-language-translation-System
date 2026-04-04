# GitHub + Render Setup Guide

Follow these steps exactly in order.

---

## Step 1 — Install Git LFS on your Mac

```bash
brew install git-lfs
git lfs install
```

---

## Step 2 — Rename your project folder

```bash
mv ~/Downloads/HandSignDetector ~/Downloads/sign-language-translator
cd ~/Downloads/sign-language-translator
```

---

## Step 3 — Replace files with the new versions

Copy these files into your project:
- `app.py` → replace existing
- `templates/index.html` → replace existing
- `requirements.txt` → new file
- `render.yaml` → new file
- `.gitignore` → new file
- `.gitattributes` → new file
- `README.md` → new file

Also rename your model label file:
```bash
mv Model/"word_label_mapping (1).pkl" Model/word_label_mapping.pkl
```

---

## Step 4 — Initialize Git repo

```bash
cd ~/Downloads/sign-language-translator
git init
git lfs track "*.h5"
git lfs track "*.pkl"
```

---

## Step 5 — Create repo on GitHub

1. Go to github.com → **New repository**
2. Name it: `sign-language-translator`
3. Set to **Public**
4. Do NOT add README (you already have one)
5. Click **Create repository**

---

## Step 6 — Push to GitHub

```bash
git add .
git commit -m "Initial commit — Sign Language Translation System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sign-language-translator.git
git push -u origin main
```

> Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 7 — Verify LFS files uploaded

On GitHub, click on `Model/word_model.h5` — it should show:

```
Stored with Git LFS
```

If it shows code instead, LFS didn't work. Run:
```bash
git lfs migrate import --include="*.h5,*.pkl"
git push --force
```

---

## Step 8 — Deploy on Render

1. Go to [render.com](https://render.com) → Sign up / Log in
2. Click **New** → **Web Service**
3. Click **Connect GitHub** → select `sign-language-translator`
4. Render reads `render.yaml` automatically
5. Click **Create Web Service**
6. Wait 5–10 minutes for build to complete
7. Your app is live at `https://sign-language-translator.onrender.com`

---

## Troubleshooting

**Models not loading on Render:**
- Check that `.h5` and `.pkl` files show "Stored with Git LFS" on GitHub
- Check Render build logs for errors

**espeak not found on Render:**
- The `render.yaml` installs it automatically via `apt-get`
- If TTS fails silently, it won't break the app — detection still works

**Camera not working on Render:**
- Render servers have no webcam — the video feed won't work on deployed version
- The app is best demonstrated locally; Render is for showing the UI and routes

**Port errors:**
- Render sets `$PORT` automatically — `app.py` reads it with `os.environ.get('PORT', 5008)`
