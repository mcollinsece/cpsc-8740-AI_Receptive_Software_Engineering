import { Amplify } from 'aws-amplify';

export const configureAmplify = () => {
  Amplify.configure({
    API: {
      GraphQL: {
        endpoint: process.env.REACT_APP_API_ENDPOINT as string,
        region: process.env.REACT_APP_REGION as string,
        defaultAuthMode: 'userPool'
      }
    },
    Auth: {
      Cognito: {
        userPoolId: process.env.REACT_APP_USER_POOL_ID as string,
        userPoolClientId: process.env.REACT_APP_USER_POOL_CLIENT_ID as string,
        signUpVerificationMethod: 'code'
      }
    }
  });
}; 