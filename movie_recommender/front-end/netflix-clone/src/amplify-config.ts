import { Amplify } from 'aws-amplify';

Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: process.env.REACT_APP_USER_POOL_ID || '',
      userPoolClientId: process.env.REACT_APP_USER_POOL_CLIENT_ID || '',
      signUpVerificationMethod: 'code',
    }
  }
});

// Validate required environment variables
const requiredEnvVars = [
  'REACT_APP_AWS_ACCESS_KEY_ID',
  'REACT_APP_AWS_SECRET_ACCESS_KEY',
  'REACT_APP_AWS_REGION',
  'REACT_APP_USER_POOL_ID',
  'REACT_APP_USER_POOL_CLIENT_ID'
];

const missingVars = requiredEnvVars.filter(
  varName => !process.env[varName]
);

if (missingVars.length > 0) {
  console.error('Missing required environment variables:', missingVars);
  throw new Error('Missing required environment variables for AWS configuration');
} 