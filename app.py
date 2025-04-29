{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyP5g4cGxORIe4ofFozpCqml",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Minitheja/HACKTHON/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 1. Install necessary packages\n",
        "!pip install transformers torch gradio nltk --quiet\n",
        "\n",
        "# 2. Import libraries\n",
        "from transformers import GPT2LMHeadModel, GPT2Tokenizer\n",
        "import torch\n",
        "import gradio as gr\n",
        "import nltk\n",
        "from nltk.sentiment import SentimentIntensityAnalyzer\n",
        "\n",
        "# 3. Download NLTK resources\n",
        "nltk.download('vader_lexicon')\n",
        "\n",
        "# 4. Initialize sentiment analyzer\n",
        "sia = SentimentIntensityAnalyzer()\n",
        "\n",
        "# 5. Load GPT-2 model and tokenizer\n",
        "model_name = \"gpt2-medium\"  # or use \"gpt2\" for faster performance\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(model_name)\n",
        "model = GPT2LMHeadModel.from_pretrained(model_name)\n",
        "\n",
        "# Fix GPT-2 padding issue\n",
        "tokenizer.pad_token = tokenizer.eos_token\n",
        "model.config.pad_token_id = model.config.eos_token_id\n",
        "\n",
        "# Move to GPU if available\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model.to(device)\n",
        "\n",
        "# Improved chatbot logic to avoid repetitive phrases and focus on the user's message\n",
        "def generate_response(user_message, chat_history):\n",
        "    # Initialize sentiment_score with a default value\n",
        "    sentiment_score = 0.0\n",
        "\n",
        "    # Avoid repetitive greetings (like \"Hi\", \"Hello\")\n",
        "    greetings = [\"hi\", \"hello\", \"hey\", \"good morning\", \"good evening\"]\n",
        "\n",
        "    # Check if the user's message is a simple greeting and return a more appropriate response\n",
        "    if any(greeting in user_message.lower() for greeting in greetings):\n",
        "        bot_reply = \"👋 Hello! How can I help you with your finances today?\"\n",
        "    else:\n",
        "        # Analyze sentiment for response tone\n",
        "        sentiment_score = sia.polarity_scores(user_message)['compound']\n",
        "\n",
        "        # Specific finance-related responses\n",
        "        if \"save money\" in user_message.lower():\n",
        "            bot_reply = \"💰 Great! Saving is a smart move. How much do you want to save each month?\"\n",
        "        elif \"invest\" in user_message.lower():\n",
        "            bot_reply = \"📈 Investing can grow your wealth. Are you looking for information on mutual funds or stocks?\"\n",
        "        elif \"budget\" in user_message.lower():\n",
        "            bot_reply = \"📊 Budgeting helps you track your expenses. Would you like tips on creating a budget?\"\n",
        "        else:\n",
        "            bot_reply = \"💬 I see you're interested in finance. Let me know your goals, and I can help guide you.\"\n",
        "\n",
        "    # Adjust tone based on sentiment score\n",
        "    if sentiment_score > 0.5:\n",
        "        prefix = \"😊 \"\n",
        "    elif sentiment_score < -0.5:\n",
        "        prefix = \"😔 \"\n",
        "    else:\n",
        "        prefix = \"💬 \"\n",
        "\n",
        "    bot_reply = prefix + bot_reply\n",
        "\n",
        "    # Add to the chat history\n",
        "    chat_history.append((user_message, bot_reply))\n",
        "    return chat_history, chat_history\n",
        "\n",
        "\n",
        "# 7. Create Gradio Interface\n",
        "with gr.Blocks() as demo:\n",
        "    gr.Markdown(\"## 💬 Financial Assistant Chatbot\")\n",
        "    chatbot = gr.Chatbot()\n",
        "    msg = gr.Textbox(label=\"Your Message\")\n",
        "    state = gr.State([])\n",
        "\n",
        "    send_btn = gr.Button(\"Send\")\n",
        "\n",
        "    send_btn.click(\n",
        "        generate_response,\n",
        "        inputs=[msg, state],\n",
        "        outputs=[chatbot, state]\n",
        "    )\n",
        "\n",
        "# 8. Launch the app\n",
        "demo.launch(debug=True)\n",
        "demo.launch(share=True)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 722
        },
        "id": "jhk0BYsbyVXf",
        "outputId": "6fff1428-96e8-41ba-f516-cf7bef48ad8e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package vader_lexicon to /root/nltk_data...\n",
            "[nltk_data]   Package vader_lexicon is already up-to-date!\n",
            "<ipython-input-7-8f76b7f3ebf5>:73: UserWarning: You have not specified a value for the `type` parameter. Defaulting to the 'tuples' format for chatbot messages, but this is deprecated and will be removed in a future version of Gradio. Please set type='messages' instead, which uses openai-style dictionaries with 'role' and 'content' keys.\n",
            "  chatbot = gr.Chatbot()\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "It looks like you are running Gradio on a hosted a Jupyter notebook. For the Gradio app to work, sharing must be enabled. Automatically setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n",
            "\n",
            "Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch().\n",
            "* Running on public URL: https://6949e0e7e704eff9b3.gradio.live\n",
            "\n",
            "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://6949e0e7e704eff9b3.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}