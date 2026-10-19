/**
 * App.jsx — Root component with routing and context provider
 */
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AssessmentProvider } from './context/AssessmentContext';

import LoginPage from './pages/LoginPage';
import LessonCompletedPage from './pages/LessonCompletedPage';
import AssessmentPage from './pages/AssessmentPage';
import ReportPage from './pages/ReportPage';
import TeacherDashboard from './pages/TeacherDashboard';
import PracticePage from './pages/PracticePage';

// Protected route: redirects to /login if no token
const PrivateRoute = ({ children }) => {
  const token = localStorage.getItem('stemToken');
  return token ? children : <Navigate to="/login" replace />;
};

import { VoiceProvider } from './context/VoiceContext';

function App() {
  return (
    <VoiceProvider>
      <AssessmentProvider>
        <BrowserRouter basename="/adaptive-quiz">
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/lesson-complete" element={<PrivateRoute><LessonCompletedPage /></PrivateRoute>} />
            <Route path="/assessment" element={<PrivateRoute><AssessmentPage /></PrivateRoute>} />
            <Route path="/practice" element={<PrivateRoute><PracticePage /></PrivateRoute>} />
            <Route path="/report" element={<PrivateRoute><ReportPage /></PrivateRoute>} />
            <Route path="/teacher" element={<TeacherDashboard />} />
            <Route path="*" element={<Navigate to="/login" replace />} />
          </Routes>
        </BrowserRouter>
      </AssessmentProvider>
    </VoiceProvider>
  );
}

export default App;
