import sys
import os

print("Python version:", sys.version)
print("Python path:", sys.path)

# Check for livekit packages
print("\nSearching for livekit related packages:")
livekit_packages = []
for path in sys.path:
    if os.path.isdir(path):
        for item in os.listdir(path):
            if item.startswith('livekit'):
                full_path = os.path.join(path, item)
                if os.path.isdir(full_path):
                    livekit_packages.append(item)
                    
print("Found livekit packages:", livekit_packages)

# Check livekit package directory structure
livekit_path = None
for path in sys.path:
    potential_path = os.path.join(path, 'livekit')
    if os.path.isdir(potential_path):
        livekit_path = potential_path
        break

if livekit_path:
    print(f"\nExamining livekit package structure at {livekit_path}:")
    for root, dirs, files in os.walk(livekit_path):
        rel_path = os.path.relpath(root, livekit_path)
        if rel_path == '.':
            print(f"Root directory: {files}")
        else:
            print(f"Subdirectory {rel_path}: {files}")

# Check livekit_agents package if exists
livekit_agents_path = None
for path in sys.path:
    potential_path = os.path.join(path, 'livekit_agents')
    if os.path.isdir(potential_path):
        livekit_agents_path = potential_path
        break

if livekit_agents_path:
    print(f"\nExamining livekit_agents package structure at {livekit_agents_path}:")
    for root, dirs, files in os.walk(livekit_agents_path):
        rel_path = os.path.relpath(root, livekit_agents_path)
        if rel_path == '.':
            print(f"Root directory: {files}")
        else:
            print(f"Subdirectory {rel_path}: {files}")
else:
    print("\nlivekit_agents package not found in Python path")

# Try to find VoiceAgent in any livekit package
print("\nSearching for VoiceAgent class in livekit packages:")
for path in sys.path:
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if 'livekit' in root and '__pycache__' not in root:
                for file in files:
                    if file.endswith('.py'):
                        try:
                            with open(os.path.join(root, file), 'r') as f:
                                content = f.read()
                                if 'class VoiceAgent' in content:
                                    print(f"Found VoiceAgent class in: {os.path.join(root, file)}")
                        except Exception as e:
                            pass  # Skip files we can't read
