import { Amplify } from 'aws-amplify';

const config = {
  Auth: {
    Cognito: {
      userPoolId: process.env.REACT_APP_USER_POOL_ID || '',
      userPoolClientId: process.env.REACT_APP_USER_POOL_CLIENT_ID || '',
      loginWith: {
        oauth: {
          domain: process.env.REACT_APP_COGNITO_DOMAIN || '',
          scopes: ['email', 'openid', 'profile'],
          redirectSignIn: [process.env.REACT_APP_REDIRECT_SIGN_IN || ''],
          redirectSignOut: [process.env.REACT_APP_REDIRECT_SIGN_OUT || ''],
          responseType: 'code' as const
        }
      }
    }
  }
};

// Validate required environment variables
const requiredEnvVars = [
  'REACT_APP_USER_POOL_ID',
  'REACT_APP_USER_POOL_CLIENT_ID',
  'REACT_APP_COGNITO_DOMAIN',
  'REACT_APP_REDIRECT_SIGN_IN',
  'REACT_APP_REDIRECT_SIGN_OUT'
];

const missingVars = requiredEnvVars.filter(
  varName => !process.env[varName]
);

if (missingVars.length > 0) {
  console.error('Missing required environment variables:', missingVars);
  throw new Error('Missing required environment variables for Cognito configuration');
}

Amplify.configure(config); 