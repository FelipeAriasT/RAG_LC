# Streamlit RAG Application

This project implements a Retrieval-Augmented Generation (RAG) workflow using Streamlit. The application allows users to interact with a chat interface to query documents and receive responses based on the retrieved information.

## Project Structure

```
streamlit-rag-app
├── app.py                # Main entry point of the Streamlit application
├── components            # Directory for reusable components
│   ├── __init__.py      # Marks the components directory as a package
│   ├── chat.py          # Chat interface for user interaction
│   ├── citations.py      # Manages the display of citations
│   ├── sidebar.py       # Sidebar for parameter selection
│   └── streaming.py     # Handles streaming of responses
├── utils                 # Directory for utility functions
│   ├── __init__.py      # Marks the utils directory as a package
│   └── helpers.py       # Helper functions for loading images and processing data
├── models                # Directory for model-related classes and functions
│   ├── __init__.py      # Marks the models directory as a package
│   └── rag_graph.py     # Implementation of the RAG graph workflow
├── config.py            # Configuration settings for the application
├── requirements.txt      # Lists dependencies required for the project
└── README.md             # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd streamlit-rag-app
pip install -r requirements.txt
```

## Usage

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

Once the application is running, you can interact with the chat interface. Use the sidebar to select a model, specify the number of results, and toggle options for query expansion and reranking.

## Features

- **Chat Interface**: Engage in a conversation with the application to ask questions and receive answers based on document retrieval.
- **Dynamic Sidebar**: Select models and configure parameters easily from the sidebar.
- **Streaming Responses**: Experience real-time feedback with loading messages during processing.
- **Citation Management**: View citations and associated images for the retrieved documents.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.