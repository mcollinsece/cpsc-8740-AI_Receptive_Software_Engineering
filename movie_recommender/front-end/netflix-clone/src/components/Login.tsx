import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import loginBg from '../assets/login_bg.png';
import yamrLogo from '../assets/yamr.png';

const LoginContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  color: white;
  position: relative;
  z-index: 2;
  
  &:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: url(${loginBg});
    background-size: cover;
    background-position: center;
    opacity: 0.5;
    z-index: -1;
  }
`;

const LoginForm = styled.form`
  background-color: rgba(0, 0, 0, 0.75);
  padding: 60px 68px;
  border-radius: 4px;
  min-width: 450px;
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

const ErrorMessage = styled.div`
  color: #e50914;
  font-size: 14px;
  margin-bottom: 16px;
`;

const Logo = styled.img`
  width: 180px;
  margin-bottom: 20px;
`;

const SignUpSection = styled.div`
  margin-top: 16px;
  color: #737373;
  font-size: 16px;
  text-align: center;
`;

const SignUpLink = styled.a`
  color: white;
  text-decoration: none;
  margin-left: 5px;
  cursor: pointer;
  
  &:hover {
    text-decoration: underline;
  }
`;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      if (!email || !password) {
        setError('Please fill in all fields');
        return;
      }

      // For demo purposes, just navigate to welcome page
      navigate('/welcome');
    } catch (error) {
      setError('Something went wrong. Please try again.');
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
          placeholder="Email"
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
          minLength={8}
        />
        <Button type="submit">
          {isSignUp ? 'Sign Up' : 'Sign In'}
        </Button>
        <SignUpSection>
          {isSignUp ? (
            <>
              Already have an account?{' '}
              <SignUpLink onClick={toggleSignUp}>
                Sign in now
              </SignUpLink>
            </>
          ) : (
            <>
              New to YAMR?{' '}
              <SignUpLink onClick={toggleSignUp}>
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