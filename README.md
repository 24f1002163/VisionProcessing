# Interactive Note Processor

A powerful AI-powered application that processes student notes images to extract key concepts, highlight them visually, and generate multilingual speech explanations.

## Features

✨ **Core Capabilities:**
- 📸 **Image Analysis**: Upload photos of student notes
- 🧠 **Concept Extraction**: Uses GPT-4o Vision to identify important concepts
- 🎨 **Visual Highlighting**: Highlights concepts with distinct colors in the image
- 🖱️ **Interactive Regions**: Click on highlighted concepts for more info
- 🎤 **Multilingual Speech**: Generate audio explanations in multiple languages
- 🌍 **Language Support**: English (US/India), Hindi, Tamil, Telugu, Marathi

## Tech Stack

**Backend:**
- Azure Functions (Python) - Serverless compute
- Microsoft Agent Framework - Agent orchestration
- GPT-4o Vision - Image analysis and concept extraction
- Azure Speech Services - Text-to-speech synthesis
- Pillow - Image processing and highlighting

**Frontend:**
- HTML5 + CSS3 + Vanilla JavaScript
- Responsive design for desktop and mobile

**Azure Services:**
- Azure OpenAI (GPT-4o)
- Azure Speech Services
- Azure Cognitive Services (Computer Vision)
- Azure Functions

## Prerequisites

- Python 3.10 or higher
- Virtual environment (`.venv`)
- Azure account with:
  - Azure OpenAI resource (GPT-4o deployment)
  - Azure Speech Services resource
  - Azure Functions resource (for deployment)
- Azure CLI (optional, for deployment)

## Environment Setup

### 1. Configure Python Environment

The virtual environment has been created with all required dependencies. Verify it's activated:

```bash
source .venv/bin/activate
```

### 2. Set Up Environment Variables

Edit the `.env` file and provide your Azure credentials:

```bash
# Azure Computer Vision (existing)
VISION_KEY=<your-vision-key>
VISION_ENDPOINT=<your-vision-endpoint>

# Azure OpenAI Configuration (for GPT-4o Vision)
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o  # or your deployment name

# Azure Speech Services
SPEECH_KEY=<your-speech-key>
SPEECH_REGION=eastus  # or your region

# Optional: For Microsoft Foundry integration
# FOUNDRY_PROJECT_ENDPOINT=<your-foundry-endpoint>
# FOUNDRY_MODEL_DEPLOYMENT_NAME=<your-model-deployment>
```

**How to Get Credentials:**
- **Azure OpenAI**: Visit Azure Portal → OpenAI → Keys and Endpoint
- **Speech Services**: Visit Azure Portal → Speech Services → Keys and Endpoint

## Project Structure

```
MS_AI_ImgProcessor/
├── concept_extraction_agent.py    # GPT-4o Vision agent for concept extraction
├── image_highlighter.py           # Image processing and highlighting logic
├── speech_generator.py            # Azure Speech Services integration
├── function_app.py               # Azure Functions HTTP endpoints
├── index.html                    # Frontend UI
├── requirements.txt              # Python dependencies
├── .env                         # Environment configuration
└── README.md                    # This file
```

## Key Modules

### `concept_extraction_agent.py`
- Uses Microsoft Agent Framework with GPT-4o Vision
- Extracts concepts from student note images
- Returns structured data with concept names, descriptions, and bounding boxes

### `image_highlighter.py`
- Processes images using Pillow
- Adds colored rectangles around extracted concepts
- Generates clickable regions for frontend interaction
- Supports multiple colors for visual distinction

### `speech_generator.py`
- Integrates Azure Speech Services (Text-to-Speech)
- Supports 6 languages: English (US/India), Hindi, Tamil, Telugu, Marathi
- Generates WAV audio in base64 format

### `function_app.py`
- Azure Functions HTTP trigger
- Three main endpoints:
  - `POST /api/upload_notes` - Process image and extract concepts
  - `POST /api/generate_speech` - Generate audio explanation
  - `GET /api/supported_languages` - Get available languages
  - `GET /api/health` - Health check

## Running Locally

### Option 1: Azure Functions Runtime (Recommended for Testing)

```bash
# Install Azure Functions Core Tools (if not already installed)
# macOS: brew tap azure/tap && brew install azure-functions-core-tools@4

# Start the local Azure Functions server
func start

# The API will be available at http://localhost:7071/api
```

### Option 2: Direct Python Execution

```bash
# For testing individual modules
python concept_extraction_agent.py
python speech_generator.py
```

## API Endpoints

### 1. Process Notes Image
**POST** `/api/upload_notes`

Request:
```json
{
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

Response:
```json
{
    "success": true,
    "highlighted_image": "iVBORw0KGgoAAAANSUhEUgAAA...",
    "concepts": [
        {
            "id": "concept_1",
            "name": "Photosynthesis",
            "description": "Process of converting sunlight into chemical energy",
            "color": {"r": 255, "g": 200, "b": 0},
            "region": {"x1": 10, "y1": 20, "x2": 150, "y2": 120},
            "category": "Biology"
        }
    ],
    "image_dimensions": {
        "width": 800,
        "height": 600
    },
    "message": "Successfully extracted 5 concepts"
}
```

### 2. Generate Speech
**POST** `/api/generate_speech`

Request:
```json
{
    "concept_name": "Photosynthesis",
    "concept_description": "Process of converting sunlight into chemical energy",
    "language": "en-US"
}
```

Response:
```json
{
    "success": true,
    "audio_base64": "UklGRiYAAABXQVZFZm10IBAAAA...",
    "audio_format": "wav",
    "language": "en-US",
    "voice": "en-US-AriaNeural",
    "concept": "Photosynthesis",
    "duration_ms": 3500
}
```

### 3. Supported Languages
**GET** `/api/supported_languages`

Response:
```json
{
    "success": true,
    "supported_languages": {
        "en-US": {
            "code": "en-US",
            "name": "English (US)",
            "voices": ["en-US-AriaNeural", "en-US-GuyNeural"],
            "default_voice": "en-US-AriaNeural"
        },
        "hi-IN": {
            "code": "hi-IN",
            "name": "Hindi",
            "voices": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
            "default_voice": "hi-IN-SwaraNeural"
        }
        // ... more languages
    }
}
```

## Frontend Usage

1. Open `index.html` in a web browser (or serve via a web server)
2. **Step 1**: Upload an image of student notes
3. **Step 2**: Click "Process Image" to extract concepts
4. **Step 3**: Select a concept from the list
5. **Step 4**: Choose a language for audio explanation
6. **Step 5**: Click "Generate Audio" to hear the explanation

## Development Workflow

### Adding New Languages

To add a new language for speech synthesis:

1. Edit `speech_generator.py`
2. Add language code to `SUPPORTED_LANGUAGES` dictionary:

```python
"pt-BR": {
    "language": "pt-BR",
    "voices": ["pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"],
    "default_voice": "pt-BR-AntonioNeural",
    "name": "Portuguese (Brazil)"
}
```

3. Verify the voice names are available in Azure Speech Services

### Customizing Highlighting Colors

Edit the `COLORS` array in `image_highlighter.py`:

```python
COLORS = [
    (255, 200, 0),      # Yellow
    (0, 200, 255),      # Cyan
    # Add more RGB tuples...
]
```

## Deployment

### Deploy to Azure Functions

```bash
# Install Azure CLI if not already installed

# Login to Azure
az login

# Create a resource group
az group create --name MyResourceGroup --location eastus

# Create a storage account (required for Functions)
az storage account create \
  --name mystorageaccount \
  --location eastus \
  --resource-group MyResourceGroup \
  --sku Standard_LRS

# Create a Function App
az functionapp create \
  --resource-group MyResourceGroup \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.13 \
  --functions-version 4 \
  --name my-note-processor \
  --storage-account mystorageaccount

# Deploy the code
func azure functionapp publish my-note-processor --build remote
```

### Deploy Frontend

1. Upload `index.html` to:
   - Azure Static Web Apps
   - Azure Blob Storage (with static site hosting)
   - Any web server (update API_BASE URL in JavaScript)

## Troubleshooting

### Issue: "SPEECH_KEY not set"
**Solution**: Add `SPEECH_KEY` and `SPEECH_REGION` to `.env` file

### Issue: "GPT-4o deployment not found"
**Solution**: Verify the deployment name matches your Azure OpenAI resource

### Issue: Image highlighting regions are incorrect
**Solution**: Ensure the image format is supported (PNG, JPG) and properly encoded in base64

### Issue: Speech generation timeout
**Solution**: Check your Speech Services region and network connectivity

## Performance Optimization

- **Image Size**: Keep input images under 20MB for faster processing
- **Caching**: Consider caching extracted concepts and generated speech
- **Async Processing**: The agent framework handles async operations automatically
- **Batch Processing**: For multiple images, use batch processing with queuing

## Security Considerations

1. **API Keys**: Never commit `.env` file to version control
2. **Authentication**: Add Azure AD or API key validation to Function endpoints
3. **Input Validation**: Validate and sanitize all user inputs
4. **CORS**: Configure CORS policies appropriately for production
5. **Rate Limiting**: Implement rate limiting on API endpoints

## Future Enhancements

- [ ] Support for video input (frame extraction)
- [ ] Handwriting recognition for better concept extraction
- [ ] PDF support for document processing
- [ ] Custom concept templates
- [ ] Concept relationship mapping (knowledge graphs)
- [ ] Export highlights and notes as PDF
- [ ] Integration with learning management systems (LMS)
- [ ] Real-time collaboration features
- [ ] Mobile app version

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Contact: support@example.com
- Documentation: https://docs.example.com

## References

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Azure Speech Services](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/)
- [Azure Functions Documentation](https://learn.microsoft.com/en-us/azure/azure-functions/)

---

**Version**: 1.0.0  
**Last Updated**: March 5, 2026
