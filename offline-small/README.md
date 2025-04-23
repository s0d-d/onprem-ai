# Offline agent with LiveKit

## Background

The goal of this project is to find out how to run voice agents in a completely offline environment. Firstly, there's a tool to build voice agents called LiveKit. And, it is best in class.
There are two kinds of agents in livekit: Multimodal, and voice-pipeline-agent. We are going for voice-pipeline-agent because we don't have any opensource mutlimodal agent yet. Multimodal means speech-to-speech agent.
In voice-pipeline-agent, we need stt, llm, and tts. And we put them in a pipeline to process human speech and answer back in synthesized speech format.

### Components:

- stt:
  implemented a custom wrapper class WhisperSTT for this. it is using faster_whisper python library (https://github.com/SYSTRAN/faster-whisper). It downloads its own model under the hood.
- llm:
  ollama serves llm model - you can choose the model depending on the hardware. llama3.1, llama3.2 were too big and the requests were timing out. using openai plugin with with_ollama method.
- tts:
  kokoro tts. using a ready-made api wrapper which serves the model.
  (also tried piper-tts, but it is problematic to install. There is no windows version. On linux, there were some dependency collisions)

### How to run

1. create venv and activate

```bash
python3 -m venv offline_venv
source offline_venv/bin/activate
```

2. install requirements
   `pip install -r requirements`
3. install ollama
   https://ollama.com/download/
4. pull suitable model
   you need models that supports tools. you can search with this link: https://ollama.com/search?c=tools
   the choose one. For my case, I chose the smallest which is:

```bash
ollama run smollm2:135m
```

but it was not good. but at least it worked.

5. install and run kokoro tts api
   for gpu:

```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest
```

for cpu:

```bash
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
```

6. download necessary files

```bash
python local.py download-files
```

7. start the agent

```bash
python local.py dev
```

## UI

You can use meet.livekit.io as UI, or run the ui locally. To run locally, clone this repo: https://github.com/livekit-examples/meet and run.

## Meeting Server

You also need to run your own turn server which organizes the meetings. To do this, refer to this: https://github.com/livekit/livekit

there is docker command:

```bash
docker run --rm -it   -p 7880:7880   -p 7881:7881   -p 5349:5349/udp   -p 5349:5349   -e LIVEKIT_KEYS="mykey: mysecret"   -e LIVEKIT_API_KEY=mykey   -e LIVEKIT_API_SECRET=mysecret   livekit/livekit-server   --dev
```

to join a meeting, you need a token. to generate:
install livekit-cli https://github.com/livekit/livekit-cli
then:

```bash
lk token create   --api-key mykey --api-secret mysecret   --join --room test_room --identity test_user   --valid-for 24h
```

also you can add a bot to a room:

```bash
lk room join --url ws://192.168.231.89:7880 --api-key mykey --api-secret mysecret --identity bot_user test_room
```

## My notes:

I first tried to run the agent locally.

- to connect to a meeting server without TLS certificate you need to host the UI in localhost. Because the browser will not allow downgrade connection: https -> ws.
- so I ran the UI locally. and, I was using WSL. I found out that WSL has a different IP than localhost.
- so I moved the UI and meeting server to windows. and tried again.
- then it started working without tls certificate.

- I also found out that the stt, llm, tts engines should be working on the machine that hosts agent. (Not the machine for meeting server).

## Current problem:

- the model is too small. if it is big, the inference time is too slow and inference request times out.
- stt accuracy can be better. (try others: Tiny.En, Base.En, Small.En, Medium.En, Large-v2, Large-v2)
- does the agent needs to something? or just answering a question is enough?

## Useful resources

- about livekit agents: https://medium.com/l7mp-technologies/running-reel-time-ai-voice-assistants-in-kubernetes-136662bd031f
- agent with search function: https://github.com/dwain-barnes/sky-livekit-agent-perplexica
- local agent but without livekit: https://github.com/amanvirparhar/weebo
- nice UI: https://github.com/livekit-examples/voice-assistant-frontend
- livekit documentation: https://docs.livekit.io/agents/
- livekit documentation for agents: https://docs.livekit.io/agents/integrations/stt/
- livekit agent examples: https://github.com/livekit/agents/tree/main/examples/voice_agents
- livekit-plugins-kokoro merge request: https://github.com/livekit/agents/pull/1615/files#
- offline whisper: https://github.com/openai/whisper/discussions/1463

# Further readings:

- https://www.home-assistant.io/blog/2022/12/20/year-of-voice/
