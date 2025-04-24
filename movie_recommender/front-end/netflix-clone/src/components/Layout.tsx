import React from 'react';
import styled from 'styled-components';
import { Outlet } from 'react-router-dom';
import loginBg from '../assets/login_bg.png';

const PageContainer = styled.div`
  min-height: 100vh;
  width: 100%;
  position: relative;
  background-image: url(${loginBg});
  background-size: cover;
  background-position: center;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
  }
`;

const Layout: React.FC = () => {
  return (
    <PageContainer>
      <Outlet />
    </PageContainer>
  );
};

export default Layout; 