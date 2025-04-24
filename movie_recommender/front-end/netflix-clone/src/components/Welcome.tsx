import React from 'react';
import styled from 'styled-components';

const WelcomeContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  color: white;
  position: relative;
  z-index: 2;
`;

const WelcomeCard = styled.div`
  background-color: rgba(0, 0, 0, 0.75);
  padding: 60px 68px;
  border-radius: 4px;
  min-width: 450px;
  text-align: center;
`;

const Title = styled.h1`
  color: white;
  font-size: 32px;
  font-weight: 500;
  margin-bottom: 28px;
`;

const Welcome: React.FC = () => {
  return (
    <WelcomeContainer>
      <WelcomeCard>
        <Title>Welcome to YAMR</Title>
      </WelcomeCard>
    </WelcomeContainer>
  );
};

export default Welcome; 