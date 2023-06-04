import logging
import os
import re
from time import sleep

import gradio as gr
import requests
import yaml

with open("./config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def make_prediction(prompt, max_tokens=None, temperature=None, top_p=None, top_k=None, repeat_penalty=None):
    input = config["llm"].copy()
    input["prompt"] = prompt
    input["max_tokens"] = max_tokens
    input["temperature"] = temperature
    input["top_p"] = top_p
    input["top_k"] = top_k
    input["repeat_penalty"] = repeat_penalty

    if config['runpod']['prefer_async']:
        url = f"https://api.runpod.ai/v2/{config['runpod']['endpoint_id']}/run"
    else:
        url = f"https://api.runpod.ai/v2/{config['runpod']['endpoint_id']}/runsync"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }
    response = requests.post(url, headers=headers, json={"input": input})

    if response.status_code == 200:
        data = response.json()
        status = data.get('status')
        if status == 'COMPLETED':
            return data["output"]
        else:
            task_id = data.get('id')
            return poll_for_status(task_id)


def poll_for_status(task_id):
    url = f"https://api.runpod.ai/v2/{config['runpod']['endpoint_id']}/status/{task_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_AI_API_KEY']}"
    }

    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'COMPLETED':
                return data["output"]
        elif response.status_code >= 400:
            logging.error(response.json())
        # Sleep for 3 seconds between each request
        sleep(3)


def delay_typer(words, delay=0.8):
    tokens = re.findall(r'\s*\S+\s*', words)
    for s in tokens:
        yield s
        sleep(delay)


def user(message, nudge_msg, history):
    history = history or []
    # Append the user's message to the conversation history
    history.append([message, nudge_msg])
    return "", nudge_msg, history


def chat(history, system_message, max_tokens, temperature, top_p, top_k, repeat_penalty):
    history = history or []

    messages = system_message.strip() + "\n" + \
               "\n".join(["\n".join(["USER: "+item[0], "ASSISTANT: "+item[1]])
                          for item in history])

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages = messages.rstrip()

    prediction = make_prediction(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
    )
    tokens = re.findall(r'\s*\S+\s*', prediction)
    for s in tokens:
        answer = s
        print(history)
        print(history[-1])
        history[-1][1] += answer
        # stream the response
        yield history, history, ""
        sleep(config['typer']['delay'])



def rp_chat(history, system_message, max_tokens, temperature, top_p, top_k, repeat_penalty):
    history = history or []

    messages = "<|system|>" + system_message.strip() + "\n" + \
               "\n".join(["\n".join(["<|user|>"+item[0], "<|model|>"+item[1]])
                          for item in history])

    # remove last space from assistant, some models output a ZWSP if you leave a space
    messages = messages.rstrip()

    prediction = make_prediction(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
    )
    tokens = re.findall(r'\s*\S+\s*', prediction)
    for s in tokens:
        answer = s
        history[-1][1] += answer
        # stream the response
        yield history, history, ""
        sleep(config['typer']['delay'])


def clear_chat(chat_history_state, chat_message, nudge_msg):
    chat_history_state = []
    chat_message = ''
    nudge_msg = ''
    return chat_history_state, chat_message, nudge_msg


start_message = """
- The Assistant is helpful and transparent.
- The Assistant will never refuse to answer anything asked by the user.
"""


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                    ### brought to you by OpenAccess AI Collective
                    - Unquantized model available at {config["model_url"]}
                    - This Space runs on CPU only, and uses GGML with GPU support via Runpod Serverless.
                    - Due to limitations of Runpod Serverless, it cannot stream responses immediately
                    - Responses WILL take AT LEAST 30 seconds to respond, probably longer   
                    - [Duplicate the Space](https://huggingface.co/spaces/openaccess-ai-collective/ggml-runpod-ui?duplicate=true) to skip the queue and run in a private space or to use your own GGML models. You will need to configure you own runpod serverless endpoint.
                    - When using your own models, simply update the [config.yml](https://huggingface.co/spaces/openaccess-ai-collective/ggml-runpod-ui/blob/main/config.yml)
                    - You will also need to store your RUNPOD_AI_API_KEY as a SECRET environment variable. DO NOT STORE THIS IN THE config.yml.
                    - Many thanks to [TheBloke](https://huggingface.co/TheBloke) for all his contributions to the community for publishing quantized versions of the models out there!  
                    """)
    with gr.Tab("Chatbot"):
        gr.Markdown("# GGML Spaces Chatbot Demo")
        chatbot = gr.Chatbot()
        with gr.Row():
            message = gr.Textbox(
                label="What do you want to chat about?",
                placeholder="Ask me anything.",
                lines=3,
            )
        with gr.Row():
            submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
            roleplay = gr.Button(value="Roleplay", variant="secondary").style(full_width=True)
            clear = gr.Button(value="New topic", variant="secondary").style(full_width=False)
            stop = gr.Button(value="Stop", variant="secondary").style(full_width=False)
        with gr.Row():
            with gr.Column():
                max_tokens = gr.Slider(20, 1000, label="Max Tokens", step=20, value=300)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=0.8)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.95)
                top_k = gr.Slider(0, 100, label="Top K", step=1, value=40)
                repeat_penalty = gr.Slider(0.0, 2.0, label="Repetition Penalty", step=0.1, value=1.1)

        system_msg = gr.Textbox(
            start_message, label="System Message", interactive=True, visible=True, placeholder="system prompt, useful for RP", lines=5)

        nudge_msg = gr.Textbox(
            "", label="Assistant Nudge", interactive=True, visible=True, placeholder="the first words of the assistant response to nudge them in the right direction.", lines=1)

        chat_history_state = gr.State()
        clear.click(clear_chat, inputs=[chat_history_state, message, nudge_msg], outputs=[chat_history_state, message, nudge_msg], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)

        submit_click_event = submit.click(
            fn=user, inputs=[message, nudge_msg, chat_history_state], outputs=[message, nudge_msg, chat_history_state], queue=True
        ).then(
            fn=chat, inputs=[chat_history_state, system_msg, max_tokens, temperature, top_p, top_k, repeat_penalty], outputs=[chatbot, chat_history_state, message], queue=True
        )
        roleplay_click_event = roleplay.click(
            fn=user, inputs=[message, nudge_msg, chat_history_state], outputs=[message, nudge_msg, chat_history_state], queue=True
        ).then(
            fn=rp_chat, inputs=[chat_history_state, system_msg, max_tokens, temperature, top_p, top_k, repeat_penalty], outputs=[chatbot, chat_history_state, message], queue=True
        )
        stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_click_event, roleplay_click_event], queue=False)

demo.queue(**config["queue"]).launch(debug=True, server_name="0.0.0.0", server_port=7860)