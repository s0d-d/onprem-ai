import subprocess

class PiperTTS:
    def __init__(self, model_path="en_US-hfc_female-medium.onnx", output_path="/tmp/output.wav"):
        self.model_path = model_path
        self.output_path = output_path

    async def synthesize(self, text: str) -> str:
        subprocess.run([
            "piper",
            "--model", self.model_path,
            "--output_file", self.output_path,
            "--text", text
        ])
        return self.output_path
