import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styled from 'styled-components';
import Login from './components/Login';
import Welcome from './components/Welcome';

const AppContainer = styled.div`
  min-height: 100vh;
  background-color: #141414;
  color: white;
`;

const App: React.FC = () => {
  return (
    <Router>
      <AppContainer>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Welcome />} />
        </Routes>
      </AppContainer>
    </Router>
  );
};

export default App;
