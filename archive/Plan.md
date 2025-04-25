Offline Components Needed

  1. STT (Speech-to-Text):
    - No built-in offline STT implementation exists
    - You'll need to create a custom plugin integrating Whisper or another offline STT model
    - Follow the STT interface from livekit.agents.stt.stt
    - Use fake_stt.py as a reference implementation
  2. TTS (Text-to-Speech):
    - No built-in offline TTS implementation exists
    - Create a custom plugin integrating Piper TTS or another offline TTS model
    - Follow the TTS interface from livekit.agents.tts.tts
    - Use fake_tts.py as a reference implementation
  3. LLM (Language Model):
    - No built-in offline LLM implementation exists
    - Create a custom plugin for Ollama to use local models
    - Follow the LLM interface from livekit.agents.llm.llm
  4. VAD (Voice Activity Detection):
    - Silero VAD is available as a ready-to-use offline component
    - Located in livekit-plugins-silero directory
    - Uses an ONNX model that runs locally

  Creating Custom Plugins

  1. Project Structure:
    - Follow the pattern in livekit-plugins-minimal
    - Create a package structure: livekit/plugins/your_plugin_name/
    - Include __init__.py, version.py, and your implementation files
  2. Plugin Registration:
    - Extend Plugin class from livekit.agents
    - Register your plugin with Plugin.register_plugin()
  3. Implementation Requirements:
    - For STT: Implement _recognize_impl method and a streaming RecognizeStream class
    - For TTS: Implement synthesize method returning a ChunkedStream and a stream method
    - For LLM: Implement chat method that handles chat context and tools
  4. Fallback Support:
    - Use FallbackAdapter to combine multiple providers
    - This allows falling back to alternative STT/TTS/LLM services when one fails

  Integration Example

  To create a fully offline voice agent:

  1. Create custom plugins for each component:
  # STT Plugin with Whisper
  class WhisperSTT(STT):
      def __init__(self, model_path):
          super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
          # Load your Whisper model here

      async def _recognize_impl(self, buffer, language, conn_options):
          # Implement transcription with Whisper

      def stream(self, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
          return WhisperRecognizeStream(self, language, conn_options)
  2. Create a voice agent using your offline components:
  async def create_agent():
      # Initialize components
      whisper_stt = WhisperSTT(model_path="path/to/whisper/model")
      piper_tts = PiperTTS(model_path="path/to/piper/model")
      ollama_llm = OllamaLLM(model="llama3")
      silero_vad = silero.VAD.load()

      # Create voice agent
      agent = VoiceAgent(
          stt=whisper_stt,
          tts=piper_tts,
          llm=ollama_llm,
          vad=silero_vad
      )

      return agent

  The repository provides all the necessary interfaces and examples to implement custom offline
  components, with the Silero VAD plugin serving as a good reference for an offline component
  implementation.
