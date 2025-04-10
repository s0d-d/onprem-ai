import sounddevice as sd

def list_devices():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']} (Input: {device['max_input_channels']}, Output: {device['max_output_channels']})")

if __name__ == "__main__":
    list_devices()
