# Netflix Clone with React and AWS Amplify

A Netflix clone built with React and AWS Amplify, featuring a responsive design and movie data from TMDB API.

## Features

- Netflix-like UI with responsive design
- Movie rows with horizontal scrolling
- Featured banner with random movie
- Movie trailers using YouTube API
- AWS Amplify integration for authentication
- TMDB API integration for movie data

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- AWS account
- TMDB API key

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   REACT_APP_TMDB_API_KEY=your_tmdb_api_key
   REACT_APP_AWS_REGION=your_aws_region
   REACT_APP_USER_POOL_ID=your_user_pool_id
   REACT_APP_USER_POOL_WEB_CLIENT_ID=your_user_pool_web_client_id
   ```

4. Start the development server:
   ```bash
   npm start
   ```

## AWS Amplify Setup

1. Create a new AWS Amplify project
2. Set up authentication using Amazon Cognito
3. Configure the environment variables in your `.env` file
4. Deploy the application using AWS Amplify Console

## Technologies Used

- React
- AWS Amplify
- Styled Components
- TMDB API
- YouTube API
- Axios

## License

MIT 