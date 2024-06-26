# Pet Care Veterinary Chatbot

## Overview
This project presents a state-of-the-art chatbot designed to assist pet owners by providing reliable health care information. The bot is built upon the powerful LangChain and LLaMA 2 models, utilizing AWS SageMaker for robust deployment and MongoDB for efficient data handling.
![web](/Web_image/web.png)
![chatbot](/Web_image/chatbot.png)

## Features
![flowchart](/Web_image/image.png)
- **LangChain Integration**: Crafting seamless conversational flows through advanced prompt engineering, ensuring precise user query understanding and response generation.
- **LLaMA 2 Model**: Utilizes the LLaMA 2 7B model from HuggingFace, offering comprehensive language understanding capabilities.
- **Scalable Deployment**: Hosting on AWS SageMaker to provide a responsive and scalable chatbot service.
- **MongoDB Atlas**: Employing MongoDB Atlas as a vector database, enhancing the chatbot's ability to fetch accurate and relevant pet health information quickly.

## Data Ingestion Pipeline
The chatbot includes a sophisticated data ingestion pipeline that leverages HuggingFace embeddings for document vectorization. This process allows the chatbot to retrieve the most pertinent information in response to user queries.

## Getting Started
To get a local copy up and running, follow these simple steps:

### Prerequisites
- AWS account with SageMaker
- MongoDB Atlas account
- HuggingFace account

### Installation
1. Clone the repo
   ```sh
   git clone git@github.com:ykj5060/Pet-Care-Web.git
2. Install required packages
   ```sh
   pip install -r requirements.txt
