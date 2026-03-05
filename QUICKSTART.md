# Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

### 1. Activate Virtual Environment
```bash
cd /Users/asha/Documents/MS_AI_ImgProcessor
source .venv/bin/activate
```

### 2. Update `.env` File
Edit the `.env` file with your Azure credentials:
```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
SPEECH_KEY=<your-speech-key>
SPEECH_REGION=eastus
```

### 3. Start the Local Server
```bash
func start
```

API runs at: `http://localhost:7071/api`

### 4. Open the Web UI
```bash
# In a new terminal
open index.html
# or use a web server
python -m http.server 8000
# Visit: http://localhost:8000
```

### 5. Try It Out!
1. Upload an image of notes
2. Click "Process Image"
3. Select a concept
4. Choose a language
5. Click "Generate Audio"

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `concept_extraction_agent.py` | GPT-4o Vision concept extraction |
| `image_highlighter.py` | Image processing and highlighting |
| `speech_generator.py` | Azure Speech Services integration |
| `function_app.py` | Azure Functions HTTP endpoints |
| `index.html` | Web UI frontend |
| `requirements.txt` | Python dependencies |
| `.env` | Configuration (Azure credentials) |

---

## 🔧 Useful Commands

### Test Individual Modules
```bash
# Test concept extraction
python concept_extraction_agent.py

# Test speech generation
python speech_generator.py
```

### View Dependencies
```bash
pip list
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Run Flask Locally (Alternative)
```bash
# If you want to use Flask instead of Azure Functions
pip install flask flask-cors
# Create a simple Flask app wrapper
```

---

## 🌐 API Endpoints

### Process Image
```bash
curl -X POST http://localhost:7071/api/upload_notes \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"iVBORw0KGg..."}'
```

### Generate Speech
```bash
curl -X POST http://localhost:7071/api/generate_speech \
  -H "Content-Type: application/json" \
  -d '{
    "concept_name":"Photosynthesis",
    "concept_description":"Converting light to energy",
    "language":"en-US"
  }'
```

### Get Languages
```bash
curl http://localhost:7071/api/supported_languages
```

---

## 🐛 Quick Fixes

### "No module named 'concept_extraction_agent'"
```bash
source .venv/bin/activate
cd /Users/asha/Documents/MS_AI_ImgProcessor
```

### "AZURE_OPENAI_ENDPOINT not set"
Make sure `.env` has all required variables

### API returning 500 errors
Check console output from `func start` for detailed error messages

### Image highlighting not working
- Verify image format (PNG/JPG)
- Check base64 encoding is correct
- Ensure concept regions are normalized (0-1)

---

## 📚 Key Language Codes

Use in the frontend language selector:

```
en-US  = English (US)
en-IN  = English (India)
hi-IN  = हिंदी (Hindi)
ta-IN  = தமிழ் (Tamil)
te-IN  = తెలుగు (Telugu)
mr-IN  = मराठी (Marathi)
```

---

## 💡 Pro Tips

1. **Faster Processing**: Use smaller, cropped images
2. **Better Results**: High-contrast, clear notes work best
3. **Language Support**: Test with all languages to verify Azure setup
4. **Caching**: Consider caching speech results
5. **Batch Processing**: Process multiple images in sequence

---

## 📖 More Documentation

- **Detailed Setup**: See `SETUP_GUIDE.md`
- **Full Documentation**: See `README.md`
- **Architecture**: See architecture diagrams in README

---

## ✅ Checklist Before Going to Production

- [ ] All Azure credentials configured
- [ ] Speech Services in correct region
- [ ] Images highlight correctly
- [ ] All languages generate audio
- [ ] Frontend works with API
- [ ] Error handling works
- [ ] Performance is acceptable
- [ ] Security best practices applied
- [ ] Logging configured
- [ ] Documentation updated

---

## 🚀 Deploy to Azure

```bash
# 1. Create resource group
az group create -n MyGroup -l eastus

# 2. Create Function App
az functionapp create \
  -g MyGroup \
  -n my-note-processor \
  -s mystorageaccount \
  --runtime python \
  --runtime-version 3.13 \
  --functions-version 4

# 3. Deploy code
func azure functionapp publish my-note-processor --build remote

# 4. Set environment variables
az functionapp config appsettings set \
  -g MyGroup \
  -n my-note-processor \
  --settings "AZURE_OPENAI_ENDPOINT=..." "AZURE_OPENAI_API_KEY=..."
```

---

## 📞 Getting Help

1. Check `SETUP_GUIDE.md` for detailed instructions
2. Review console output for error messages
3. Verify Azure credentials in `.env`
4. Test individual modules with Python
5. Check network tab in browser DevTools

---

**Everything installed and ready!** 🎉
