import axiosInstance from './axiosInstance';
import { getTeacherKey } from '../utils/teacherAuth';

export const generateQuestionsFromPDF = async ({ pdf, lessonId, conceptTag, numQuestions }) => {
  const formData = new FormData();
  formData.append('pdf', pdf);
  formData.append('lessonId', lessonId);
  formData.append('conceptTag', conceptTag || 'General');
  formData.append('numQuestions', String(numQuestions || 9));

  const res = await axiosInstance.post('/questions/generate', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
      'X-Teacher-Key': getTeacherKey(),
    },
    timeout: 660_000,
  });
  return res.data;
};

// ── Teacher review: generated questions stay "pending" until approved ─────────
const teacherHeaders = () => ({ headers: { 'X-Teacher-Key': getTeacherKey() } });

export const getPendingQuestions = async (lessonId) => {
  const res = await axiosInstance.get('/questions/review', {
    ...teacherHeaders(),
    params: lessonId ? { lessonId } : {},
  });
  return res.data;
};

export const updateQuestion = async (questionId, changes) => {
  const res = await axiosInstance.put(`/questions/${encodeURIComponent(questionId)}`, changes, teacherHeaders());
  return res.data;
};

export const approveQuestions = async (questionIds) => {
  const res = await axiosInstance.post('/questions/approve', { questionIds }, teacherHeaders());
  return res.data;
};

export const rejectQuestion = async (questionId) => {
  const res = await axiosInstance.delete(`/questions/${encodeURIComponent(questionId)}`, teacherHeaders());
  return res.data;
};

// Student self-study: generate ungraded practice questions from their own PDF.
// Not saved to the graded question bank, not scored, not tracked in reports.
export const generatePracticeQuestionsFromPDF = async ({ pdf, conceptTag, numQuestions }) => {
  const formData = new FormData();
  formData.append('pdf', pdf);
  formData.append('conceptTag', conceptTag || 'General');
  formData.append('numQuestions', String(numQuestions || 9));

  const res = await axiosInstance.post('/questions/practice-generate', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 660_000,
  });
  return res.data;
};
