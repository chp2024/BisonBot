# BisonBot: Your Howard Information Guide

## Overview
BisonBot is an AI-powered chatbot designed to enhance the user experience on Howard University's website. By addressing the challenges of navigation and accessibility, BisonBot provides a frictionless interface for students, faculty, and visitors to access information efficiently.

Currently, the website looks like this:
<img width="1449" alt="Screenshot 2024-04-28 at 2 36 01 AM" src="https://github.com/chp2024/BisonBot/assets/64405568/fbbbc8f7-ffa4-4dd0-bdda-301ce1430e50">

With the current setup, the bot is only accessible on a few web pages. But, with BisonBot we aim to make the feature available in a separate browser so users can ask Howard related questions and get responses. 
<img width="817" alt="Screenshot 2024-04-26 at 6 29 25 PM-2" src="https://github.com/chp2024/BisonBot/assets/64405568/2560aad2-302e-4182-ae05-12215a306161">

## Features
- **Authentication**: Users can sign in using GitHub and/or Google.
- **Prompt Handling**: Intuitive processing of user queries with response generation tailored to Howard's spirit.
- **Accessibility**: Toggle between light and dark themes, and read-aloud options.

## Requirements
- Login and logout capabilities
- Responsive chat with history viewing and new tab initiation
- Accessibility features to support diverse user needs

## Technologies Used
- **GitHub OAuth**: Ensures secure user authentication.
- **Python and OpenAI API**: For backend development and chat response handling.
- **Chainlit**: Enhance the front end and support real-time interactions.

## Additional Features
- **Third-Party Logins**: Integration with platforms like Google and GitHub.
- **User Identification**: Custom responses based on user roles (students, parents, faculty)
- **BisonBot Icon**: Quick access chat window on all web pages.

## Database Design
- **Chat History Storage**: Managed via Literal AI API.
- **Vector Database**: Chroma DB stores vector embeddings for efficient retrieval.
- **User Information Storage**: Managed by ChainLit’s Database.

## User Interface
- Integration with GitHub/Google for login.
- Real-time chat interface with options for attachments and copying text.
- Toggle for light/dark mode.

## Testing Strategy
- **Predefined Benchmarks**: Evaluate responses for accuracy, relevance, and tone.
- **Prompt Variability Testing**: Ensure diverse and contextually appropriate responses.
- **Content Generation and Retrieval**: Test for factual accuracy and up-to-date information.

## Quick Start Guide
1. Clone the repository and navigate to the project directory.
2. Install dependencies: `pip install -r requirements.txt`.
3. Access BisonBot through localhost on your browser.
