const awsmobile = {
    "aws_project_region": "us-east-1",
    "aws_cognito_region": "us-east-1",
    "aws_user_pools_id": process.env.REACT_APP_USER_POOL_ID,
    "aws_user_pools_web_client_id": process.env.REACT_APP_USER_POOL_CLIENT_ID,
    "oauth": {
        "domain": process.env.REACT_APP_COGNITO_DOMAIN,
        "scope": [
            "phone",
            "email",
            "openid",
            "profile",
            "aws.cognito.signin.user.admin"
        ],
        "redirectSignIn": process.env.REACT_APP_REDIRECT_SIGN_IN,
        "redirectSignOut": process.env.REACT_APP_REDIRECT_SIGN_OUT,
        "responseType": "code"
    },
    "federationTarget": "COGNITO_USER_POOLS",
    "aws_appsync_graphqlEndpoint": process.env.REACT_APP_GRAPHQL_ENDPOINT,
    "aws_appsync_region": "us-east-1",
    "aws_appsync_authenticationType": "AMAZON_COGNITO_USER_POOLS",
    "aws_cognito_username_attributes": ["email"],
    "aws_cognito_signup_attributes": ["email"],
    "aws_cognito_mfa_configuration": "OFF",
    "aws_cognito_mfa_types": [],
    "aws_cognito_password_protection_settings": {
        "passwordPolicyMinLength": 8,
        "passwordPolicyCharacters": []
    },
    "aws_cognito_verification_mechanisms": ["email"]
};

export default awsmobile; 