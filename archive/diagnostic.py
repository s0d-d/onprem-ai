#!/usr/bin/env python3
"""
Diagnostic script to identify the correct import path for LiveKit VoiceAgent
"""
import sys
import os
import pkgutil
import importlib

print("===== LiveKit Agents Diagnostic Tool =====")
print(f"Python version: {sys.version}")
print("\n=== Installed LiveKit Packages ===")

# Check for installed livekit packages
def check_installed_packages():
    try:
        import livekit
        print("✓ livekit package is installed")
    except ImportError:
        print("✕ livekit package is not installed")
    
    try:
        import livekit.agents
        print("✓ livekit.agents subpackage exists")
    except ImportError:
        print("✕ livekit.agents subpackage does not exist")
    
    try:
        import livekit_agents
        print("✓ livekit_agents package is installed")
    except ImportError:
        print("✕ livekit_agents package is not installed")

check_installed_packages()

print("\n=== Looking for VoiceAgent class ===")

# Check all possible locations for VoiceAgent
def find_voice_agent():
    possible_locations = [
        'livekit.agents',
        'livekit.agents.voice',
        'livekit.voice',
        'livekit_agents',
        'livekit_agents.voice',
    ]
    
    found = False
    
    for location in possible_locations:
        try:
            module = importlib.import_module(location)
            if hasattr(module, 'VoiceAgent'):
                print(f"✓ Found VoiceAgent in {location}")
                found = True
                print(f"  Class name: {module.VoiceAgent.__name__}")
                print(f"  Module path: {module.__file__}")
                return location
            else:
                print(f"✕ No VoiceAgent in {location}")
                # Check what classes are available in this module
                attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                if attrs:
                    print(f"  Available classes: {', '.join(attrs[:10])}")
                    if 'Voice' in attrs:
                        print(f"  Note: Found 'Voice' which might be similar to 'VoiceAgent'")
        except ImportError:
            print(f"✕ Could not import {location}")
    
    if not found:
        print("\nVoiceAgent class not found in common locations.")
    
    return None

agent_location = find_voice_agent()

print("\n=== Examining livekit.agents Module Structure ===")

# Check module structure
def examine_module_structure(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"Module path: {module.__file__}")
        
        # Get all submodules
        path = module.__path__
        submodules = [name for _, name, _ in pkgutil.iter_modules(path)]
        print(f"Submodules: {', '.join(submodules) if submodules else 'None'}")
        
        # Check main exports
        exports = [attr for attr in dir(module) if not attr.startswith('_')]
        print(f"Exports: {', '.join(exports[:10]) + '...' if len(exports) > 10 else ', '.join(exports)}")
        
        return submodules
    except ImportError:
        print(f"Could not import {module_name}")
        return []

agents_submodules = examine_module_structure('livekit.agents')

print("\n=== Recommended Import Statement ===")

if agent_location:
    print(f"from {agent_location} import VoiceAgent, VoiceAgentOptions, VoiceAgentAnalytics")
else:
    # Check for 'voice' module which might contain VoiceAgent
    if 'voice' in agents_submodules:
        print("# Try importing from voice submodule:")
        print("from livekit.agents.voice import VoiceAgent, VoiceAgentOptions, VoiceAgentAnalytics")
    else:
        print("# Could not determine the correct import path.")
        print("# Check livekit-agents documentation for your specific version (1.0.0rc9)")
        print("# Or try one of these:")
        print("# from livekit.agents.voice import VoiceAgent")
        print("# from livekit.agents import Voice as VoiceAgent")

print("\n=== Checking for Common Classes ===")

# Look for any classes with similar names
voice_classes = []

def find_voice_related_classes(module_name):
    try:
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            if 'Voice' in attr_name or 'Agent' in attr_name:
                voice_classes.append((module_name, attr_name))
    except ImportError:
        pass

find_voice_related_classes('livekit.agents')
for submodule in agents_submodules:
    find_voice_related_classes(f'livekit.agents.{submodule}')

if voice_classes:
    print("Found these voice-related classes:")
    for module, class_name in voice_classes:
        print(f"from {module} import {class_name}")
else:
    print("No voice-related classes found.")

print("\n=== Package Version Information ===")

try:
    import livekit
    print(f"livekit version: {livekit.__version__ if hasattr(livekit, '__version__') else 'unknown'}")
except ImportError:
    print("livekit package not found")

try:
    import pkg_resources
    livekit_agents_version = pkg_resources.get_distribution("livekit-agents").version
    print(f"livekit-agents version: {livekit_agents_version}")
except (ImportError, pkg_resources.DistributionNotFound):
    print("Could not determine livekit-agents version")

print("\n=== Complete ===")
print("Run this script to identify the correct import path for VoiceAgent")
