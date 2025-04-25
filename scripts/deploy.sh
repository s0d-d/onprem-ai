#!/bin/bash
# Script to deploy the offline voice agent to Kubernetes

# Set variables
NAMESPACE="offline-voice-agent"
IMAGE_NAME="voice-agent"
IMAGE_TAG="latest"

# Build Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Create namespace
echo "Creating namespace..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes-manifests.yaml

# Create token for testing
echo "Creating a test token for the LiveKit room..."
echo "This token will be valid for 24 hours."
echo "Wait for the LiveKit server pod to be running first."
