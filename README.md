🧠 NeuroSenseAI
Multimodal Emotion Intelligence using Deep Learning, Attention & Graph Neural Networks
📌 Overview

NeuroSenseAI is an advanced multimodal emotion recognition system that intelligently understands human emotions by jointly analyzing:

🧍 Facial expressions

🎙️ Speech signals

📝 Textual content

Unlike traditional single-modal approaches, NeuroSenseAI leverages attention-based fusion and Graph Neural Networks (GNNs) to perform context-aware and relational emotion reasoning, making it robust, scalable, and highly expressive.

🚀 Key Highlights

🔹 Multimodal emotion understanding (Face + Speech + Text)

🔹 Attention-based fusion to dynamically weight modalities

🔹 Graph Neural Network for relational reasoning

🔹 Modular, extensible deep learning architecture

🔹 Interactive Streamlit demo

🔹 Hackathon & major-project ready

🏗️ System Architecture
Face CNN Embeddings  ┐
                     ├─ Attention Fusion ─ GNN ─ Classifier ─ Emotion Output
Speech CNN+BiLSTM    ┤
                     │
Text BERT Embeddings ┘

🔑 Core Ideas

Attention Fusion learns the importance of each modality dynamically.

Graph Neural Network models relationships between samples instead of treating emotions independently.

🧠 Technologies Used

Deep Learning: CNN, BiLSTM, BERT

Multimodal Fusion: Attention Mechanism

Graph Learning: Graph Convolutional Networks (GCN)

Frameworks: PyTorch, Transformers

Demo UI: Streamlit

Language: Python

📁 Project Structure
NeuroSenseAI/
│
├── demo/
│   └── app.py                # Streamlit demo
│
├── face_model.py             # Face emotion model
├── speech_model.py           # Speech emotion model
├── text_model.py             # BERT-based text model
│
├── fusion_attention.py       # Attention-based fusion
├── fusion_gnn.py             # Graph Neural Network
├── final_model.py            # End-to-end multimodal model
│
├── test_text_inference.py    # CLI testing script
├── requirements.txt
├── README.md
└── .gitignore


⚠️ Datasets, embeddings, and trained weights are intentionally excluded for repository cleanliness.

▶️ Demo (Streamlit)

Run the interactive demo locally:

pip install -r requirements.txt
streamlit run demo/app.py

Demo Features

Text-based emotion prediction

Confidence score display

Clean and fast UI

💡 Face and speech modalities are supported internally and can be added to the demo later.

🧪 Testing

Run a quick CLI test for text emotion inference:

python test_text_inference.py

📊 Emotion Classes

The system predicts one of the following 7 emotions:

Angry 😠

Fear 😨

Happy 😊

Sad 😢

Neutral 😐

Surprised 😲

Calm 😌

🎯 Use Cases

Mental health monitoring

Human–computer interaction

Sentiment-aware chatbots

Smart surveillance systems

Assistive technologies

🔮 Future Enhancements

🎥 Real-time webcam facial emotion detection

🎧 Audio emotion recognition from microphone

🌐 Cloud deployment (AWS / Hugging Face Spaces)

📱 Mobile-friendly UI

🔁 Online learning with feedback loops

👨‍💻 Author

Lalit
B.Tech – Artificial Intelligence & Data Science
📌 Passionate about Deep Learning, Multimodal AI & Intelligent Systems

⭐ If you like this project

Give it a ⭐ on GitHub — it really helps!