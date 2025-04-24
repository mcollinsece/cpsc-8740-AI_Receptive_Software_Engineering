import React, { useState } from 'react';
import styled from 'styled-components';
import { useNavigate } from 'react-router-dom';
import yamrLogo from '../assets/yamr.png';
import { signIn, signUp } from 'aws-amplify/auth';

const LoginContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  color: white;
  position: relative;
  z-index: 2;
`;

const LoginForm = styled.form`
  background-color: rgba(0, 0, 0, 0.75);
  padding: 60px 68px;
  border-radius: 4px;
  min-width: 450px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const Logo = styled.img`
  width: 180px;
  margin-bottom: 20px;
`;

const Title = styled.h1`
  color: white;
  font-size: 32px;
  font-weight: 500;
  margin-bottom: 28px;
`;

const Input = styled.input`
  width: 100%;
  padding: 16px;
  margin-bottom: 16px;
  border-radius: 4px;
  border: none;
  background-color: #333;
  color: white;
  font-size: 16px;

  &::placeholder {
    color: #8c8c8c;
  }
`;

const Button = styled.button`
  width: 100%;
  padding: 16px;
  background-color: #e50914;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  margin-top: 24px;

  &:hover {
    background-color: #f40612;
  }
`;

const SignUpSection = styled.div`
  margin-top: 16px;
  color: #737373;
  font-size: 16px;
`;

const SignUpLink = styled.a`
  color: white;
  text-decoration: none;
  margin-left: 5px;
  
  &:hover {
    text-decoration: underline;
  }
`;

const ErrorMessage = styled.div`
  color: #e50914;
  font-size: 14px;
  margin-bottom: 16px;
`;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isSignUp) {
        // Sign up logic with auto sign in
        await signUp({
          username: email,
          password,
          options: {
            userAttributes: {
              email,
            },
            autoSignIn: {
              enabled: true
            }
          },
        });
        // Auto sign in after successful signup
        await signIn({
          username: email,
          password,
        });
        navigate('/welcome');
      } else {
        // Sign in logic
        await signIn({
          username: email,
          password,
        });
        navigate('/welcome');
      }
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleSignUp = () => {
    setIsSignUp(!isSignUp);
    setError('');
  };

  return (
    <LoginContainer>
      <LoginForm onSubmit={handleSubmit}>
        <Logo src={yamrLogo} alt="YAMR Logo" />
        <Title>{isSignUp ? 'Sign Up' : 'Sign In'}</Title>
        {error && <ErrorMessage>{error}</ErrorMessage>}
        <Input
          type="email"
          placeholder="Email or phone number"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <Button type="submit" disabled={loading}>
          {loading ? 'Loading...' : (isSignUp ? 'Sign Up' : 'Sign In')}
        </Button>
        <SignUpSection>
          {isSignUp ? (
            <>
              Already have an account?{' '}
              <SignUpLink href="#" onClick={toggleSignUp}>
                Sign in now
              </SignUpLink>
            </>
          ) : (
            <>
              New to YAMR?{' '}
              <SignUpLink href="#" onClick={toggleSignUp}>
                Sign up now
              </SignUpLink>
            </>
          )}
        </SignUpSection>
      </LoginForm>
    </LoginContainer>
  );
};

export default Login; 